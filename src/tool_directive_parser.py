#!/usr/bin/env python3
"""
Tool Directive Parser for nbscribe
Parses AI responses to extract and validate tool directives for notebook editing
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolDirective:
    """Represents a parsed tool directive from AI response"""
    tool: str  # insert_cell, edit_cell, delete_cell
    code: str  # The code content (can be empty for delete_cell)
    pos: Optional[int] = None  # Position for insert_cell (fallback)
    cell_id: Optional[str] = None  # Cell ID for edit_cell/delete_cell
    before: Optional[str] = None  # Insert before this cell ID
    after: Optional[str] = None   # Insert after this cell ID
    language: str = "python"  # Code language
    raw_match: str = ""  # Original matched text for debugging


class ToolDirectiveParser:
    """
    Parser for extracting and validating tool directives from AI responses.
    
    Expected format:
    ```python
    code here
    ```
    
    ```
    TOOL: insert_cell
    POS: 3
    ```
    """
    
    def __init__(self):
        # Pattern to match code blocks followed by tool directives
        self.code_directive_pattern = re.compile(
            r'```(\w+)?\s*\n(.*?)\n```\s*\n\s*```\s*\n(.*?)\n```',
            re.DOTALL | re.MULTILINE
        )
        
        # Pattern to match standalone tool directives (e.g., for delete_cell)
        self.standalone_directive_pattern = re.compile(
            r'```\s*\n(.*?)\n```',
            re.DOTALL | re.MULTILINE
        )
        
        # Pattern for tool directive content
        self.tool_pattern = re.compile(r'TOOL:\s*(\w+)', re.IGNORECASE)
        self.pos_pattern = re.compile(r'POS:\s*(\d+)', re.IGNORECASE)
        self.cell_id_pattern = re.compile(r'CELL_ID:\s*(\S+)', re.IGNORECASE)
        self.before_pattern = re.compile(r'BEFORE:\s*(\S+)', re.IGNORECASE)
        self.after_pattern = re.compile(r'AFTER:\s*(\S+)', re.IGNORECASE)
    
    def parse_response(self, response_text: str) -> List[ToolDirective]:
        """
        Parse AI response text to extract tool directives.
        
        Args:
            response_text: The AI response text
            
        Returns:
            List of ToolDirective objects
        """
        directives = []
        
        # First, find all code block + directive pairs
        code_matches = self.code_directive_pattern.findall(response_text)
        
        for match in code_matches:
            language = match[0] or "python"
            code = match[1].strip()
            directive_text = match[2].strip()
            
            # Debug logging for newline investigation
            logger.info(f"DIRECTIVE PARSER - Raw code length: {len(match[1])}")
            logger.info(f"DIRECTIVE PARSER - Stripped code length: {len(code)}")
            logger.info(f"DIRECTIVE PARSER - Code contains \\n: {'\\n' in code}")
            logger.info(f"DIRECTIVE PARSER - Code repr: {repr(code[:100])}")
            
            # Parse the directive
            directive = self._parse_directive(language, code, directive_text)
            if directive:
                directive.raw_match = f"```{language}\n{code}\n```\n```\n{directive_text}\n```"
                directives.append(directive)
                logger.info(f"Parsed tool directive: {directive.tool}")
        
        # Then find standalone directives (for delete_cell, etc.)
        # Remove already matched code+directive blocks first
        remaining_text = self.code_directive_pattern.sub('', response_text)
        
        standalone_matches = self.standalone_directive_pattern.findall(remaining_text)
        for directive_text in standalone_matches:
            directive_text = directive_text.strip()
            
            # Check if this looks like a tool directive
            if self.tool_pattern.search(directive_text):
                directive = self._parse_directive("", "", directive_text)
                if directive:
                    directive.raw_match = f"```\n{directive_text}\n```"
                    directives.append(directive)
                    logger.info(f"Parsed standalone tool directive: {directive.tool}")
        
        return directives
    
    def _parse_directive(self, language: str, code: str, directive_text: str) -> Optional[ToolDirective]:
        """Parse a single tool directive block"""
        try:
            # Extract tool type
            tool_match = self.tool_pattern.search(directive_text)
            if not tool_match:
                return None
            
            tool = tool_match.group(1).lower()
            
            # Validate tool type
            if tool not in ['insert_cell', 'edit_cell', 'delete_cell']:
                logger.warning(f"Unknown tool type: {tool}")
                return None
            
            # Extract position (for insert_cell fallback)
            pos = None
            pos_match = self.pos_pattern.search(directive_text)
            if pos_match:
                pos = int(pos_match.group(1))
            
            # Extract cell ID (for edit_cell/delete_cell)
            cell_id = None
            cell_id_match = self.cell_id_pattern.search(directive_text)
            if cell_id_match:
                cell_id = cell_id_match.group(1)
            
            # Extract BEFORE/AFTER (for insert_cell)
            before = None
            before_match = self.before_pattern.search(directive_text)
            if before_match:
                before = before_match.group(1)
                
            after = None
            after_match = self.after_pattern.search(directive_text)
            if after_match:
                after = after_match.group(1)
            
            # Validate required parameters
            if tool == 'insert_cell':
                has_relative = before or after
                has_absolute = pos is not None
                
                if not has_relative and not has_absolute:
                    logger.warning(f"insert_cell requires BEFORE, AFTER, or POS parameter")
                    return None
                    
                # Can't have both relative and absolute
                if has_relative and has_absolute:
                    logger.warning(f"insert_cell cannot have both relative (BEFORE/AFTER) and absolute (POS) positioning")
                    return None
                    
                # Can't have both BEFORE and AFTER
                if before and after:
                    logger.warning(f"insert_cell cannot have both BEFORE and AFTER")
                    return None
            
            if tool in ['edit_cell', 'delete_cell'] and cell_id is None:
                logger.warning(f"{tool} requires CELL_ID parameter")
                return None
            
            return ToolDirective(
                tool=tool,
                code=code,
                pos=pos,
                cell_id=cell_id,
                before=before,
                after=after,
                language=language
            )
            
        except Exception as e:
            logger.error(f"Error parsing directive: {e}")
            return None
    
    def validate_directive(self, directive: ToolDirective) -> bool:
        """Validate a tool directive for safety and correctness"""
        try:
            # Basic validation
            if not directive.tool:
                return False
            
            # Tool-specific validation
            if directive.tool == 'insert_cell':
                has_relative = directive.before or directive.after
                has_absolute = directive.pos is not None and directive.pos >= 0
                
                if not has_relative and not has_absolute:
                    logger.warning("insert_cell requires BEFORE, AFTER, or POS parameter")
                    return False
                    
                # Can't have both relative and absolute positioning
                if has_relative and has_absolute:
                    logger.warning("insert_cell cannot have both relative and absolute positioning")
                    return False
                    
                # Can't have both BEFORE and AFTER
                if directive.before and directive.after:
                    logger.warning("insert_cell cannot have both BEFORE and AFTER")
                    return False
                
                if not directive.code.strip():
                    logger.warning("insert_cell with empty code")
                    return False
            
            elif directive.tool == 'edit_cell':
                if not directive.cell_id:
                    return False
                if not directive.code.strip():
                    logger.warning("edit_cell with empty code")
                    return False
            
            elif directive.tool == 'delete_cell':
                if not directive.cell_id:
                    return False
                # delete_cell can have empty code
            
            # Code safety checks (basic)
            if directive.code and directive.language == 'python':
                # Basic safety - no dangerous imports
                dangerous_patterns = [
                    'import os',
                    'import subprocess',
                    'import sys',
                    '__import__',
                    'exec(',
                    'eval(',
                ]
                
                code_lower = directive.code.lower()
                for pattern in dangerous_patterns:
                    if pattern in code_lower:
                        logger.warning(f"Potentially dangerous code detected: {pattern}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating directive: {e}")
            return False


def test_parser():
    """Simple test for the tool directive parser"""
    parser = ToolDirectiveParser()
    
    # Test case 1: insert_cell
    test_response_1 = """
Here's a cell to calculate primes:

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(2, 100) if is_prime(n)]
print(primes)
```

```
TOOL: insert_cell
POS: 3
```

This will add the prime calculation at position 3.
"""
    
    directives = parser.parse_response(test_response_1)
    assert len(directives) == 1
    assert directives[0].tool == 'insert_cell'
    assert directives[0].pos == 3
    assert 'is_prime' in directives[0].code
    assert parser.validate_directive(directives[0])
    
    print("✅ Test 1 passed: insert_cell directive")
    
    # Test case 2: Multiple directives with BEFORE/AFTER
    test_response_2 = """
I'll add a plot and update the title:

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.show()
```

```
TOOL: insert_cell
AFTER: data-loading-cell
```

And update the existing cell:

```python
plt.title("Updated Plot Title")
```

```
TOOL: edit_cell
CELL_ID: plot_cell_123
```
"""
    
    directives = parser.parse_response(test_response_2)
    assert len(directives) == 2
    assert directives[0].tool == 'insert_cell'
    assert directives[0].after == 'data-loading-cell'
    assert directives[0].pos is None
    assert directives[1].tool == 'edit_cell'
    assert directives[1].cell_id == 'plot_cell_123'
    
    print("✅ Test 2 passed: Multiple directives with BEFORE/AFTER")
    
    # Test case 3: delete_cell
    test_response_3 = """
I'll remove that debugging cell:

```
TOOL: delete_cell
CELL_ID: debug_print_cell
```
"""
    
    directives = parser.parse_response(test_response_3)
    assert len(directives) == 1
    assert directives[0].tool == 'delete_cell'
    assert directives[0].cell_id == 'debug_print_cell'
    assert directives[0].code == ''
    
    print("✅ Test 3 passed: delete_cell directive")
    
    # Test case 4: BEFORE directive
    test_response_4 = """
Let's add setup code before the main analysis:

```python
import pandas as pd
import numpy as np
```

```
TOOL: insert_cell
BEFORE: analysis-cell-456
```
"""
    
    directives = parser.parse_response(test_response_4)
    assert len(directives) == 1
    assert directives[0].tool == 'insert_cell'
    assert directives[0].before == 'analysis-cell-456'
    assert directives[0].after is None
    assert directives[0].pos is None
    assert parser.validate_directive(directives[0])
    
    print("✅ Test 4 passed: BEFORE directive")
    
    print("🎉 All tool directive parser tests passed!")


if __name__ == "__main__":
    # Run tests
    test_parser() 