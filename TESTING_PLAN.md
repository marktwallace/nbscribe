# 🧪 nbscribe Notebook Modification Testing Plan

## Overview
Testing the newly implemented notebook modification features including cell insertion, editing, deletion, and AI context awareness.

---

## Phase 1: Environment Setup

### Commands to run:
```bash
cd /Users/mark.wallace/src/nbscribe
conda activate nbscribe
python main.py
```

### ✅ Results:
- [ ] Server started successfully on port 5317
- [ ] No error messages in terminal
- [ ] Can access http://localhost:5317

**Notes:**

---

## Phase 2: Create Test Notebook

### Steps:
1. Navigate to http://localhost:5317
2. Click "Create New Notebook" or use interface to create notebook
3. Name it `test_notebook.ipynb`
4. Manually add these cells in Jupyter:
   - **Cell 1**: `import pandas as pd\nimport numpy as np`
   - **Cell 2**: `df = pd.read_csv("data.csv")\nprint(df.head())`
5. Save the notebook (Ctrl+S in Jupyter)

### ✅ Results:
- [ ] Notebook created successfully
- [ ] Both cells added and saved
- [ ] Can see notebook in file listing

**Notes:**

---

## Phase 3: Cell ID Analysis (Exit & Restart Test)

### Steps:
1. Exit nbscribe server (Ctrl+C)
2. Open `test_notebook.ipynb` in text editor
3. Look for `"metadata": {"id": "..."}` in each cell
4. Record the cell IDs for reference
5. Restart nbscribe: `python main.py`
6. Navigate to: `http://localhost:5317/notebook/test_notebook.ipynb`

### ✅ Results:
- [ ] Cell IDs were automatically added to JSON
- [ ] Cell IDs are UUIDs (long random strings)
- [ ] Split-pane interface loads correctly
- [ ] Session ID shows notebook pattern

**Cell IDs found:**
- Cell 1 (imports): `_____________________`
- Cell 2 (data loading): `_____________________`

**Notes:**

---

## Phase 4: AI Context Awareness

### Test 4.1: Basic Context Check
**Ask AI:** "What cells do you see in my notebook?"

### ✅ Expected:
- AI describes 2 cells with content previews
- AI mentions available cell IDs
- AI knows this is `test_notebook.ipynb`

### ✅ Results:
- [ ] AI saw correct number of cells
- [ ] AI provided accurate content previews  
- [ ] AI mentioned cell IDs for referencing

**AI Response Summary:**

**Notes:**

---

## Phase 5: Cell Insertion Tests

### Test 5.1: Insert at Beginning (POS)
**Ask AI:** "Add a cell at the beginning to import matplotlib"

### ✅ Expected:
- AI suggests `TOOL: insert_cell` with `POS: 0`
- Apply button appears below code block
- Clicking Apply inserts cell at top

### ✅ Results:
- [ ] AI suggested correct tool directive
- [ ] Apply button worked
- [ ] Cell appeared at correct position in Jupyter
- [ ] Cell contains matplotlib import

**Directive Generated:**

**Notes:**

### Test 5.2: Insert AFTER Existing Cell  
**Ask AI:** "Add a data analysis cell after the data loading cell"

### ✅ Expected:
- AI uses `AFTER: [actual-cell-id]` with Cell 2's ID
- New cell appears between Cell 2 and end

### ✅ Results:
- [ ] AI used correct AFTER positioning
- [ ] AI referenced correct cell ID
- [ ] Cell inserted in right location

**Directive Generated:**

**Notes:**

### Test 5.3: Insert BEFORE Existing Cell
**Ask AI:** "Add a data cleaning step before the analysis cell"

### ✅ Expected:
- AI uses `BEFORE: [cell-id]` with analysis cell ID  
- New cell appears in correct position

### ✅ Results:
- [ ] AI used correct BEFORE positioning
- [ ] Cell inserted in right location

**Directive Generated:**

**Notes:**

---

## Phase 6: Cell Editing Tests

### Test 6.1: Edit Existing Cell
**Ask AI:** "Change the first cell to also import seaborn"

### ✅ Expected:
- AI suggests `TOOL: edit_cell` with correct `CELL_ID`
- Apply updates cell content
- Cell outputs are cleared

### ✅ Results:
- [ ] AI identified correct cell to edit
- [ ] Edit applied successfully
- [ ] Content updated in Jupyter
- [ ] Outputs cleared

**Directive Generated:**

**Notes:**

### Test 6.2: Edit Non-Existent Cell (Error Test)
Create this directive manually in chat:

```python
print("test")
```

```
TOOL: edit_cell
CELL_ID: fake-cell-123
```

### ✅ Expected:
- Error message: "Cell ID 'fake-cell-123' not found"
- No notebook corruption

### ✅ Results:
- [ ] Appropriate error message shown
- [ ] Notebook remained intact

**Error Message:**

**Notes:**

---

## Phase 7: Cell Deletion Tests

### Test 7.1: Delete Existing Cell
**Ask AI:** "Remove the data loading cell"

### ✅ Expected:
- AI suggests `delete_cell` with correct cell ID
- Cell disappears from notebook

### ✅ Results:
- [ ] AI identified correct cell
- [ ] Cell deleted successfully
- [ ] Notebook structure updated

**Directive Generated:**

**Notes:**

---

## Phase 8: Empty Notebook Test

### Steps:
1. Create new empty notebook: `empty_test.ipynb`
2. Navigate to: `http://localhost:5317/notebook/empty_test.ipynb`
3. **Ask AI:** "Add the first cell to import pandas"

### ✅ Expected:
- AI uses `POS: 0` for empty notebook
- Cell insertion works correctly

### ✅ Results:
- [ ] AI handled empty notebook correctly
- [ ] Used POS positioning as fallback
- [ ] First cell created successfully

**Notes:**

---

## Phase 9: Session Context Tests

### Test 9.1: Multiple Notebook Sessions
1. Open `test_notebook.ipynb` in Tab 1
2. Open `empty_test.ipynb` in Tab 2  
3. Apply directives in each tab

### ✅ Expected:
- Each session works with correct notebook
- Session IDs contain notebook names
- No cross-contamination

### ✅ Results:
- [ ] Session isolation working
- [ ] Correct notebook targeted in each tab

**Session IDs:**
- Tab 1: `_____________________`
- Tab 2: `_____________________`

**Notes:**

### Test 9.2: Chat-Only Session  
1. Navigate to: `http://localhost:5317/chat`
2. Try asking for notebook modifications

### ✅ Expected:
- AI mentions no notebook is open
- Directives fail gracefully

### ✅ Results:
- [ ] AI handled no-notebook case appropriately
- [ ] Error messages were user-friendly

**Notes:**

---

## Phase 10: UX Experience Tests

### Test 10.1: Live Updates
Apply several directives and observe:

### ✅ UX Elements to Check:
- [ ] Jupyter iframe updates immediately after Apply
- [ ] Success messages appear in chat
- [ ] Apply buttons become disabled after use
- [ ] Status messages replace buttons
- [ ] Loading indicators work properly

**UX Notes:**

### Test 10.2: Page Refresh Persistence
1. Apply several directives
2. Refresh the page
3. Check conversation history

### ✅ Expected:
- All notebook changes persist
- Chat history shows all directives and results

### ✅ Results:
- [ ] Changes persisted after refresh
- [ ] Chat history intact

**Notes:**

### Test 10.3: System Prompt Transparency
Click "System Prompt (for transparency)" expander

### ✅ Expected:
- System prompt displays
- Contains notebook modification instructions

### ✅ Results:
- [ ] System prompt accessible
- [ ] Contains relevant instructions

**Notes:**

---

## Phase 11: Complex Workflow Test

### Challenge: Create a Complete Data Analysis Workflow
**Ask AI:** "Help me create a complete data analysis workflow in this empty notebook"

### ✅ Goals:
- AI suggests multiple cells with proper positioning
- Apply suggestions in sequence
- Build logical notebook structure

### ✅ Results:
- [ ] AI provided comprehensive workflow
- [ ] Positioning suggestions were logical
- [ ] Final notebook has good structure

**Workflow Created:**
1. 
2. 
3. 
4. 
5. 

**Notes:**

---

## 🏁 Final Assessment

### Overall UX Rating: ⭐⭐⭐⭐⭐ ( /5 stars)

### What Worked Well:
- 
- 
- 

### Issues Encountered:
- 
- 
- 

### Suggestions for Improvement:
- 
- 
- 

### Ready for Production Use? 
- [ ] Yes, with current features
- [ ] Yes, after addressing issues above
- [ ] No, needs more work

**Final Notes:** 