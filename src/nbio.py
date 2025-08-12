"""
Notebook I/O Module for htmx-based nbscribe

Direct nbformat I/O operations without Jupyter server dependency.
Handles .ipynb files with cell ID management and basic validation.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import nbformat
from nbformat.validator import ValidationError

logger = logging.getLogger(__name__)


def read_nb(path: str) -> Dict[str, Any]:
    """
    Read notebook from file and ensure all cells have stable IDs.
    
    Args:
        path: Path to .ipynb file
        
    Returns:
        Notebook dictionary with guaranteed cell IDs
        
    Raises:
        FileNotFoundError: If notebook file doesn't exist
        ValidationError: If notebook format is invalid
    """
    notebook_path = Path(path)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb_dict = json.load(f)
        
        # Validate notebook format
        try:
            nbformat.validate(nb_dict)
        except ValidationError as e:
            logger.warning(f"Notebook validation warning for {path}: {e}")
            # Continue anyway for compatibility
        
        # Ensure all cells have IDs
        ensure_cell_ids(nb_dict)
        
        logger.info(f"Read notebook {path} with {len(nb_dict.get('cells', []))} cells")
        return nb_dict
        
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in notebook {path}: {e}")
    except Exception as e:
        raise ValidationError(f"Failed to read notebook {path}: {e}")


def write_nb(path: str, nb: Dict[str, Any]) -> None:
    """
    Write notebook to file with validation and backup.
    
    Args:
        path: Path to .ipynb file
        nb: Notebook dictionary to write
        
    Raises:
        ValidationError: If notebook format is invalid
        IOError: If file cannot be written
    """
    # Validate notebook before writing
    try:
        nbformat.validate(nb)
    except ValidationError as e:
        logger.error(f"Notebook validation failed: {e}")
        raise
    
    # Ensure all cells have IDs before writing
    ensure_cell_ids(nb)
    
    notebook_path = Path(path)
    
    # Create backup if file exists
    if notebook_path.exists():
        backup_path = notebook_path.with_suffix('.ipynb.bak')
        try:
            backup_path.write_text(notebook_path.read_text(encoding='utf-8'), encoding='utf-8')
            logger.debug(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    # Write notebook
    try:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        logger.info(f"Wrote notebook {path} with {len(nb.get('cells', []))} cells")
        
    except Exception as e:
        raise IOError(f"Failed to write notebook {path}: {e}")


def ensure_cell_ids(nb: Dict[str, Any]) -> None:
    """
    Ensure all cells have stable UUIDs.
    
    Modifies notebook in-place by adding IDs where missing.
    Uses short hex format compatible with nbformat expectations.
    
    Args:
        nb: Notebook dictionary to modify
    """
    cells = nb.get("cells", [])
    
    for cell in cells:
        # Check if cell already has a valid top-level ID
        if "id" not in cell or not cell["id"]:
            # Generate a new ID in nbformat format (short hex string)
            cell["id"] = str(uuid.uuid4()).replace("-", "")[:8]
        
        # Ensure metadata exists
        if "metadata" not in cell:
            cell["metadata"] = {}
        
        # Sync ID to metadata for backward compatibility
        cell["metadata"]["id"] = cell["id"]
    
    logger.debug(f"Ensured cell IDs for {len(cells)} cells")


def find_cell_index(nb: Dict[str, Any], cell_id: str) -> Optional[int]:
    """
    Find the index of a cell by its ID.
    
    Checks both top-level 'id' and metadata.id locations for compatibility.
    
    Args:
        nb: Notebook dictionary
        cell_id: Cell ID to search for
        
    Returns:
        Cell index (0-based) or None if not found
    """
    cells = nb.get("cells", [])
    
    for i, cell in enumerate(cells):
        # Check for ID in multiple locations
        current_id = cell.get("id") or cell.get("metadata", {}).get("id")
        if current_id == cell_id:
            return i
    
    return None


def get_cell_by_id(nb: Dict[str, Any], cell_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a cell by its ID.
    
    Args:
        nb: Notebook dictionary
        cell_id: Cell ID to search for
        
    Returns:
        Cell dictionary or None if not found
    """
    index = find_cell_index(nb, cell_id)
    if index is not None:
        return nb["cells"][index]
    return None


def insert_cell(nb: Dict[str, Any], cell: Dict[str, Any], index: int) -> None:
    """
    Insert a cell at the specified index.
    
    Args:
        nb: Notebook dictionary to modify
        cell: Cell dictionary to insert
        index: Position to insert at (0-based)
        
    Raises:
        IndexError: If index is out of bounds
    """
    cells = nb.get("cells", [])
    
    if index < 0 or index > len(cells):
        raise IndexError(f"Index {index} out of bounds for {len(cells)} cells")
    
    # Ensure cell has an ID
    if "id" not in cell or not cell["id"]:
        cell["id"] = str(uuid.uuid4()).replace("-", "")[:8]
    
    # Ensure metadata
    if "metadata" not in cell:
        cell["metadata"] = {}
    cell["metadata"]["id"] = cell["id"]
    
    cells.insert(index, cell)
    logger.debug(f"Inserted cell {cell['id']} at index {index}")


def delete_cell(nb: Dict[str, Any], cell_id: str) -> bool:
    """
    Delete a cell by its ID.
    
    Args:
        nb: Notebook dictionary to modify
        cell_id: Cell ID to delete
        
    Returns:
        True if cell was found and deleted, False otherwise
    """
    index = find_cell_index(nb, cell_id)
    if index is not None:
        removed_cell = nb["cells"].pop(index)
        logger.debug(f"Deleted cell {cell_id} from index {index}")
        return True
    
    logger.warning(f"Cell {cell_id} not found for deletion")
    return False


def update_cell_source(nb: Dict[str, Any], cell_id: str, source: str) -> bool:
    """
    Update the source code of a cell.
    
    Args:
        nb: Notebook dictionary to modify
        cell_id: Cell ID to update
        source: New source code
        
    Returns:
        True if cell was found and updated, False otherwise
    """
    cell = get_cell_by_id(nb, cell_id)
    if cell is None:
        logger.warning(f"Cell {cell_id} not found for source update")
        return False
    
    # Convert source to list format (nbformat standard)
    if isinstance(source, str):
        # Split into lines preserving newlines
        lines = source.split('\n')
        source_lines = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1 and line == "":
                # Skip empty last line (result of trailing \n)
                continue
            elif i == len(lines) - 1:
                # Last non-empty line doesn't get \n
                source_lines.append(line)
            else:
                # All other lines get \n
                source_lines.append(line + "\n")
    else:
        source_lines = source
    
    cell["source"] = source_lines
    
    # Clear outputs for code cells when editing
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    
    logger.debug(f"Updated source for cell {cell_id}")
    return True


def create_empty_notebook() -> Dict[str, Any]:
    """
    Create a new empty notebook with proper structure.
    
    Returns:
        Empty notebook dictionary
    """
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "cells": []
    }


def create_code_cell(source: str = "", cell_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new code cell.
    
    Args:
        source: Cell source code
        cell_id: Optional cell ID (generates one if not provided)
        
    Returns:
        Code cell dictionary
    """
    if cell_id is None:
        cell_id = str(uuid.uuid4()).replace("-", "")[:8]
    
    # Convert source to proper format
    if isinstance(source, str):
        lines = source.split('\n')
        source_lines = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1 and line == "":
                continue
            elif i == len(lines) - 1:
                source_lines.append(line)
            else:
                source_lines.append(line + "\n")
    else:
        source_lines = source
    
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {"id": cell_id},
        "source": source_lines,
        "execution_count": None,
        "outputs": []
    }


def create_markdown_cell(source: str = "", cell_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new markdown cell.
    
    Args:
        source: Cell source markdown
        cell_id: Optional cell ID (generates one if not provided)
        
    Returns:
        Markdown cell dictionary
    """
    if cell_id is None:
        cell_id = str(uuid.uuid4()).replace("-", "")[:8]
    
    # Convert source to proper format
    if isinstance(source, str):
        lines = source.split('\n')
        source_lines = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1 and line == "":
                continue
            elif i == len(lines) - 1:
                source_lines.append(line)
            else:
                source_lines.append(line + "\n")
    else:
        source_lines = source
    
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"id": cell_id},
        "source": source_lines
    }
