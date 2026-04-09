import os
import configparser

# ROOT_DIR = tools/ (one above src/)
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_SRC_DIR)

INI_PATH = os.path.join(_ROOT_DIR, "pac_browser.ini")


def load_settings() -> dict:
    cfg = configparser.ConfigParser()
    if os.path.exists(INI_PATH):
        cfg.read(INI_PATH)
    return dict(cfg["pac_browser"]) if "pac_browser" in cfg else {}


def save_settings(**kwargs):
    cfg = configparser.ConfigParser()
    if os.path.exists(INI_PATH):
        cfg.read(INI_PATH)
    if "pac_browser" not in cfg:
        cfg["pac_browser"] = {}
    for k, v in kwargs.items():
        cfg["pac_browser"][k] = str(v)
    with open(INI_PATH, "w") as f:
        cfg.write(f)


def validate_game_dir(path: str) -> bool:
    """Check that the path looks like a Crimson Desert install."""
    return os.path.isfile(os.path.join(path, "0009", "0.pamt"))


DARK_STYLE = """
* {
    font-family: "Segoe UI", sans-serif;
}
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #d4d4e0;
}
QMenuBar {
    background-color: #14142a;
    color: #b0b0c0;
    border-bottom: 1px solid #2a2a42;
    padding: 2px 0;
    font-size: 13px;
}
QMenuBar::item {
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #2e2e4a;
}
QMenu {
    background-color: #1e1e38;
    color: #d4d4e0;
    border: 1px solid #2e2e48;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #3a3a5c;
}
QMenu::separator {
    height: 1px;
    background: #2e2e48;
    margin: 4px 8px;
}
QLineEdit {
    background-color: #22223a;
    color: #e0e0ec;
    border: 1px solid #33334d;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: #4a6fa5;
}
QLineEdit:focus {
    border-color: #5b8def;
}
QListView {
    background-color: #1a1a30;
    color: #c8c8d8;
    border: 1px solid #2a2a42;
    border-radius: 6px;
    outline: none;
    font-size: 12px;
    padding: 2px;
}
QListView::item {
    padding: 4px 8px;
    border-radius: 3px;
}
QListView::item:selected {
    background-color: #2e4a7a;
    color: #ffffff;
}
QListView::item:hover:!selected {
    background-color: #22223a;
}
QSplitter::handle {
    background-color: #2a2a42;
}
QSplitter::handle:horizontal {
    width: 2px;
}
QStatusBar {
    background-color: #14142a;
    color: #7878a0;
    border-top: 1px solid #2a2a42;
    font-size: 12px;
}
QStatusBar::item {
    border: none;
}
QPushButton {
    background-color: #2e4a7a;
    color: #e8e8f0;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #3a5a90;
}
QPushButton:pressed {
    background-color: #243d65;
}
QPushButton:disabled {
    background-color: #22223a;
    color: #555568;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #3a3a55;
    min-height: 30px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #4a4a68;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #3a3a55;
    min-width: 30px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #4a4a68;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
    width: 0;
}
QLabel#countLabel {
    color: #5a5a78;
    font-size: 11px;
    padding: 0 4px;
}
QLabel#infoStrip {
    color: #8888a0;
    background-color: #14142a;
    border-top: 1px solid #2a2a42;
    padding: 4px 12px;
    font-size: 12px;
}
QLabel#loadingLabel {
    color: #6a6a88;
    font-size: 16px;
}
QLabel#setupTitle {
    color: #e0e0f0;
    font-size: 24px;
    font-weight: bold;
}
QLabel#setupHint {
    color: #7878a0;
    font-size: 14px;
}
QLabel#setupError {
    color: #cc4444;
    font-size: 13px;
}
QComboBox {
    background-color: #22223a;
    color: #e0e0ec;
    border: 1px solid #33334d;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    min-width: 80px;
}
QComboBox:hover {
    border-color: #5b8def;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 20px;
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #1e1e38;
    color: #d4d4e0;
    border: 1px solid #2e2e48;
    selection-background-color: #2e4a7a;
    outline: none;
}
QMessageBox {
    background-color: #1e1e38;
}
QMessageBox QLabel {
    color: #d4d4e0;
}
"""
