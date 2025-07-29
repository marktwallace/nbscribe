"""
AI Tool Directive Parser

Parses markdown responses from AI to detect tool directives and inject approval buttons.
Format: code block followed immediately by TOOL metadata block.

Example:
```python
print("hello")
```

```
TOOL: insert_cell
POS: 3
```
"""

import re
import html
import json
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ToolDirective:
    """Represents a parsed tool directive"""
    
    def __init__(self, tool: str, pos: Optional[int] = None, cell_id: Optional[str] = None, 
                 code: str = "", language: str = "python"):
        self.tool = tool
        self.pos = pos
        self.cell_id = cell_id
        self.code = code
        self.language = language
        self.directive_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for this directive"""
        import time
        return f"directive_{int(time.time() * 1000)}_{hash(self.code) % 10000}"
    
    def validate(self) -> bool:
        """Validate that directive has required fields"""
        if self.tool not in ["insert_cell", "edit_cell", "delete_cell"]:
            return False
        
        if self.tool == "insert_cell" and self.pos is None:
            return False
            
        if self.tool in ["edit_cell", "delete_cell"] and not self.cell_id:
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.directive_id,
            "tool": self.tool,
            "pos": self.pos,
            "cell_id": self.cell_id,
            "code": self.code,
            "language": self.language
        }

class ToolDirectiveParser:
    """Parses markdown content and injects tool directive approval buttons"""
    
    def __init__(self):
        # Pattern to match code block followed by tool directive
        self.pattern = re.compile(
            r'```(\w+)?\n(.*?)\n```\s*\n\s*```\s*\n(.*?)\n```',
            re.DOTALL | re.MULTILINE
        )
    
    def parse_markdown(self, markdown_text: str) -> Tuple[str, List[ToolDirective]]:
        """
        Parse markdown and return modified HTML with buttons + list of directives
        
        Returns:
            Tuple of (modified_html, list_of_directives)
        """
        directives = []
        
        def replace_directive(match):
            language = match.group(1) or "python"
            code = match.group(2).strip()
            metadata = match.group(3).strip()
            
            # Parse the metadata
            directive = self._parse_metadata(metadata, code, language)
            
            if directive and directive.validate():
                directives.append(directive)
                return self._create_directive_html(directive, code, language)
            else:
                # Malformed directive - squawk loudly
                error_msg = f"❌ MALFORMED TOOL DIRECTIVE: {metadata}"
                logger.error(f"Invalid tool directive: {metadata}")
                return self._create_error_html(code, language, error_msg)
        
        # Replace all matches
        modified_text = self.pattern.sub(replace_directive, markdown_text)
        
        return modified_text, directives
    
    def _parse_metadata(self, metadata: str, code: str, language: str) -> Optional[ToolDirective]:
        """Parse tool metadata block"""
        try:
            lines = [line.strip() for line in metadata.split('\n') if line.strip()]
            
            tool = None
            pos = None
            cell_id = None
            
            for line in lines:
                if line.startswith('TOOL:'):
                    tool = line.split(':', 1)[1].strip()
                elif line.startswith('POS:'):
                    pos = int(line.split(':', 1)[1].strip())
                elif line.startswith('CELL_ID:'):
                    cell_id = line.split(':', 1)[1].strip()
            
            if not tool:
                return None
                
            return ToolDirective(tool, pos, cell_id, code, language)
            
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            return None
    
    def _create_directive_html(self, directive: ToolDirective, code: str, language: str) -> str:
        """Create HTML for code block with approval buttons"""
        escaped_code = html.escape(code)
        
        # Create the code block
        code_html = f'<pre><code class="language-{language}">{escaped_code}</code></pre>'
        
        # Create approval buttons
        buttons_html = self._create_buttons_html(directive)
        
        # Wrap in container
        return f'''
<div class="tool-directive-container" data-directive-id="{directive.directive_id}">
    {code_html}
    {buttons_html}
</div>
'''
    
    def _create_buttons_html(self, directive: ToolDirective) -> str:
        """Create approval buttons HTML"""
        # Generate button text based on tool type
        if directive.tool == "insert_cell":
            approve_text = f"✅ Insert Cell"
            if directive.pos is not None:
                approve_text += f" (pos {directive.pos})"
        elif directive.tool == "edit_cell":
            approve_text = f"✅ Edit Cell"
            if directive.cell_id:
                approve_text += f" ({directive.cell_id})"
        elif directive.tool == "delete_cell":
            approve_text = f"✅ Delete Cell"
            if directive.cell_id:
                approve_text += f" ({directive.cell_id})"
        else:
            approve_text = "✅ Apply"
        
        directive_json = html.escape(json.dumps(directive.to_dict()))
        
        return f'''
<div class="tool-directive-buttons">
    <button class="approve-btn" 
            onclick="approveDirective('{directive.directive_id}')" 
            data-directive='{directive_json}'>
        {approve_text}
    </button>
    <button class="reject-btn" 
            onclick="rejectDirective('{directive.directive_id}')">
        ❌ Reject
    </button>
</div>
'''
    
    def _create_error_html(self, code: str, language: str, error_msg: str) -> str:
        """Create HTML for malformed directive"""
        escaped_code = html.escape(code)
        
        return f'''
<div class="tool-directive-error">
    <pre><code class="language-{language}">{escaped_code}</code></pre>
    <div class="error-message">{error_msg}</div>
</div>
'''
    
    def mark_directive_applied(self, directive_id: str, success: bool, result_msg: str = "") -> str:
        """Generate HTML for applied/rejected directive state"""
        if success:
            status_class = "directive-applied"
            status_text = f"✅ Applied"
            if result_msg:
                status_text += f": {result_msg}"
        else:
            status_class = "directive-rejected" 
            status_text = f"❌ Rejected"
            if result_msg:
                status_text += f": {result_msg}"
        
        return f'<div class="tool-directive-status {status_class}">{status_text}</div>'

# Singleton instance
parser = ToolDirectiveParser()

def parse_tool_directives(markdown_text: str) -> Tuple[str, List[ToolDirective]]:
    """Parse tool directives in markdown text"""
    return parser.parse_markdown(markdown_text)

def create_directive_status(directive_id: str, success: bool, result_msg: str = "") -> str:
    """Create status HTML for completed directive"""
    return parser.mark_directive_applied(directive_id, success, result_msg) 