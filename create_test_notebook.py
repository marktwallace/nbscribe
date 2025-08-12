#!/usr/bin/env python3
"""Create a test notebook for htmx migration testing"""

from src import nbio

def main():
    # Create a test notebook
    nb = nbio.create_empty_notebook()

    # Add test cells
    cell1 = nbio.create_code_cell("print('Hello from htmx nbscribe!')\nimport sys\nprint(f'Python version: {sys.version}')")

    cell2 = nbio.create_code_cell("# Test basic math\nresult = 2 + 2\nprint(f'2 + 2 = {result}')\nresult")

    cell3 = nbio.create_markdown_cell("# Test Markdown Cell\n\nThis is a test markdown cell for the htmx interface.")

    nbio.insert_cell(nb, cell1, 0)
    nbio.insert_cell(nb, cell2, 1)
    nbio.insert_cell(nb, cell3, 2)

    # Save the notebook
    nbio.write_nb('test_notebook.ipynb', nb)
    print('Test notebook created successfully!')

if __name__ == "__main__":
    main()
