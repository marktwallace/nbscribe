#!/usr/bin/env python3
"""
Test script for tool directive parsing
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tool_directive_parser import parse_tool_directives

def test_parser():
    """Test the tool directive parser with sample inputs"""
    
    print("🧪 Testing Tool Directive Parser...")
    
    # Test 1: Insert cell
    test1 = """Here's how to print hello world:

```python
print("Hello, World!")
```

```
TOOL: insert_cell
POS: 3
```

This will create a new cell at position 3."""
    
    print("\n1. Testing insert_cell directive...")
    html1, directives1 = parse_tool_directives(test1)
    print(f"   Found {len(directives1)} directive(s)")
    if directives1:
        print(f"   Tool: {directives1[0].tool}, POS: {directives1[0].pos}")
        print(f"   Valid: {directives1[0].validate()}")
    
    # Test 2: Edit cell
    test2 = """Let's update the plotting code:

```python
import matplotlib.pyplot as plt
# Added comment here
plt.plot(x, y)
plt.show()
```

```
TOOL: edit_cell
CELL_ID: abc123
```"""
    
    print("\n2. Testing edit_cell directive...")
    html2, directives2 = parse_tool_directives(test2)
    print(f"   Found {len(directives2)} directive(s)")
    if directives2:
        print(f"   Tool: {directives2[0].tool}, CELL_ID: {directives2[0].cell_id}")
        print(f"   Valid: {directives2[0].validate()}")
    
    # Test 3: Multiple directives
    test3 = """Here are two changes:

First, add a new import:

```python
import pandas as pd
```

```
TOOL: insert_cell
POS: 1
```

Then update the existing cell:

```python
df = pd.read_csv('data.csv')
print(df.head())
```

```
TOOL: edit_cell
CELL_ID: data_cell
```"""
    
    print("\n3. Testing multiple directives...")
    html3, directives3 = parse_tool_directives(test3)
    print(f"   Found {len(directives3)} directive(s)")
    for i, directive in enumerate(directives3):
        print(f"   Directive {i+1}: {directive.tool}, Valid: {directive.validate()}")
    
    # Test 4: Malformed directive
    test4 = """This is a broken directive:

```python
print("test")
```

```
INVALID_TOOL: broken
```"""
    
    print("\n4. Testing malformed directive...")
    html4, directives4 = parse_tool_directives(test4)
    print(f"   Found {len(directives4)} directive(s)")
    print(f"   HTML contains error: {'MALFORMED' in html4}")
    
    print("\n✅ Parser tests completed!")
    
    # Show sample HTML output
    print("\n📄 Sample HTML output for test 1:")
    print(html1[:200] + "..." if len(html1) > 200 else html1)

if __name__ == "__main__":
    test_parser() 