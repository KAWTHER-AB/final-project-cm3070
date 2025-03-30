from IPython.display import display, HTML

def styled_print(text):
    """
    Nicely formats section headers for display in both Jupyter notebooks and terminal.
    """
    try:
        display(HTML(f"<span style='color: lightblue; font-size: 16px; font-weight: bold;'>{text}</span>"))
    except:
        print("\n" + text)
