"""
Exercise to learn about recursive funcions

Given a sudoku puzzle, the script returns the solution
if the puzzle has one
"""

# pylint: disable=too-many-return-statements
def get_square(i: int, j: int) -> list:
    """
    Get the indexes numbers for the corresponding square
    """
    square_1 = [(r, c) for r in range(3) for c in range(3)]
    square_2 = [(r, c) for r in range(3) for c in range(3, 6)]
    square_3 = [(r, c) for r in range(3) for c in range(6, 9)]
    square_4 = [(r, c) for r in range(3, 6) for c in range(3)]
    square_5 = [(r, c) for r in range(3, 6) for c in range(3, 6)]
    square_6 = [(r, c) for r in range(3, 6) for c in range(6, 9)]
    square_7 = [(r, c) for r in range(6, 9) for c in range(3)]
    square_8 = [(r, c) for r in range(6, 9) for c in range(3, 6)]
    square_9 = [(r, c) for r in range(6, 9) for c in range(6, 9)]

    if (i, j) in square_1:
        return square_1
    elif (i, j) in square_2:
        return square_2
    elif (i, j) in square_3:
        return square_3
    elif (i, j) in square_4:
        return square_4
    elif (i, j) in square_5:
        return square_5
    elif (i, j) in square_6:
        return square_6
    elif (i, j) in square_7:
        return square_7
    elif (i, j) in square_8:
        return square_8
    else:
        return square_9


def is_available_number(table: list, i: int, j: int, number: int) -> bool:
    """
    Return True if numbar is an available number to use, else False
    """
    row_numbers = table[i]
    column_numbers = [table[row][j] for row in range(len(table))]
    square_numbers = [table[r][c] for (r, c) in get_square(i, j)]

    return bool(number not in set(row_numbers + column_numbers + square_numbers))


def find_empty_positions(table: list) -> tuple:
    """
    Return a tuple with three elements true and the coordinates if there is an empty element
    or False if there aren't empty spaces
    """
    # pylint: disable=consider-using-enumerate
    for row in range(len(table)):
        for column in range(len(table)):
            if table[row][column] == 0:
                return (True, row, column)
    return False, None, None


def sudoku_solve(table: list) -> bool:
    """
    Solve the sudoku problem with backtracking algorithm
    """

    # Checks If there's no empty positions
    empty_check, row, column = find_empty_positions(table)
    if not empty_check:
        # No empty positions, so puzzle solved
        return True

    # Backtracking algorithm
    # Iterate trough all nine numbers
    for number in range(1, 10):

        # checks if number is available to use.
        if is_available_number(table, row, column, number):

            table[row][column] = number

            if sudoku_solve(table):
                return True

            table[row][column] = 0

    # No number available to use, so there's no solution
    return False


def print_table(table: list) -> None:
    """
    Prints the table formatted
    """
    counter = 1
    for row in table:
        if counter % 3 != 0 or counter == 9:
            print(
                str(row[0])
                + " "
                + str(row[1])
                + " "
                + str(row[2])
                + " | "
                + str(row[3])
                + " "
                + str(row[4])
                + " "
                + str(row[5])
                + " | "
                + str(row[6])
                + " "
                + str(row[7])
                + " "
                + str(row[8])
            )
        else:
            print(
                str(row[0])
                + " "
                + str(row[1])
                + " "
                + str(row[2])
                + " | "
                + str(row[3])
                + " "
                + str(row[4])
                + " "
                + str(row[5])
                + " | "
                + str(row[6])
                + " "
                + str(row[7])
                + " "
                + str(row[8])
                + "\n---------------------"
            )
        counter += 1


def main():
    """
    Run the sudoku solver
    """
    table = [
        [0, 0, 0, 7, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 4, 6],
        [0, 0, 0, 6, 0, 3, 0, 0, 8],
        [0, 6, 7, 5, 0, 0, 4, 9, 2],
        [2, 0, 0, 9, 0, 0, 0, 7, 0],
        [0, 8, 5, 0, 0, 0, 6, 0, 3],
        [0, 0, 0, 2, 0, 4, 0, 0, 0],
        [0, 2, 3, 0, 6, 5, 0, 0, 0],
        [5, 0, 8, 3, 0, 0, 0, 6, 0],
    ]

    if sudoku_solve(table):
        print_table(table)
    else:
        print("There is no solution")


if __name__ == "__main__":
    # Execute main function
    main()
