import cv2

Y1, Y2 = 50, 112            # timestamp rows
X_OFF, X_END = 36, 613      # timestamp columns
CELL_W = 23.6               # width of one character (monospaced font)
W = 24                      # fixed crop width per character
MARGIN = 3                  # extra pixels searched each side
DATE_LEFT = 51              # left edge of "MM-DD-YYYY"
TIME_LEFT = 408             # left edge of "HH:MM:SS"
DATE_CELLS = [0, 1, 3, 4, 6, 7, 8, 9]   # digits only, skip '-'
TIME_CELLS = [0, 1, 3, 4, 6, 7]         # digits only, skip ':'


def cell_x_positions():
    xs = [DATE_LEFT + i * CELL_W for i in DATE_CELLS]
    xs += [TIME_LEFT + i * CELL_W for i in TIME_CELLS]
    return [int(round(x)) - X_OFF for x in xs]       # 14 positions


def get_cells(img):
    """Return (tight, wide): 14 binary crops each (8 date digits, 6 time digits)."""
    gray = cv2.cvtColor(img[Y1:Y2, X_OFF:X_END], cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tight, wide = [], []
    for x1 in cell_x_positions():
        tight.append(binary[:, x1:x1 + W])
        wide.append(binary[:, x1 - MARGIN:x1 + W + MARGIN])
    return tight, wide