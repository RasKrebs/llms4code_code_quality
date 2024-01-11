#!/usr/bin/env python

"""
A script for parsing a flake8 error log and generating useful stats about the
errors in the code.
Author: Ben Lopatin (I think I wrote it, at least, no guarantee)
License: BSD
"""

import sys
from collections import OrderedDict


ERROR_CODE_GROUPS = {
    "E1": "Indentation",
    "E2": "Whitespace",
    "E3": "Blank line",
    "E4": "Import",
    "E5": "Line length",
    "E7": "Statement",
    "E9": "Runtime",
    "W1": "Indentation warning",
    "W2": "Whitespace warning",
    "W3": "Blank line warning",
    "W6": "Deprecation warning",
    "C9": "Excessive complexity",
    "N8": "Naming error",
    "F4": "Import warning",
    "F8": "Undefined or unused name",
}


ERROR_CODES = {
    # PEP8
    "E273": "tab after keyword",
    "E224": "tab after operator",
    "E274": "tab before keyword",
    "E223": "tab before operator",
    "W291": "trailing whitespace",
    "E201": "whitespace after '('",
    "E202": "whitespace before ')'",
    "E203": "whitespace before ':'",
    "E211": "whitespace before '('",
    "E113": "unexpected indentation",
    "E303": "too many blank lines (3)",
    "W191": "indentation contains tabs",
    "W292": "no newline at end of file",
    "W391": "blank line at end of file",
    "E112": "expected an indented block",
    "E231": "missing whitespace after ','",
    "E401": "multiple imports on one line",
    "W603": "'<>' is deprecated, use '!='",
    "E271": "multiple spaces after keyword",
    "E222": "multiple spaces after operator",
    "E272": "multiple spaces before keyword",
    "E301": "expected 1 blank line, found 0",
    "W293": "blank line contains whitespace",
    "E221": "multiple spaces before operator",
    "E302": "expected 2 blank lines, found 0",
    "E703": "statement ends with a semicolon",
    "E901": "SyntaxError or IndentationError",
    "E116": "unexpected indentation (comment)",
    "E225": "missing whitespace around operator",
    "W601": ".has_key() is deprecated, use 'in'",
    "E115": "expected an indented block (comment)",
    "E265": "block comment should start with '# '",
    "W602": "deprecated form of raising exception",
    "E111": "indentation is not a multiple of four",
    "E262": "inline comment should start with '# '",
    "E704": "multiple statements on one line (def)",
    "E266": "too many leading '#' for block comment",
    "E713": "test for membership should be 'not in'",
    "W604": "backticks are deprecated, use 'repr()'",
    "E701": "multiple statements on one line (colon)",
    "E721": "do not compare types, use 'isinstance()'",
    "E228": "missing whitespace around modulo operator",
    "E261": "at least two spaces before inline comment",
    "E101": "indentation contains mixed spaces and tabs",
    "E304": "blank lines found after function decorator",
    "E502": "the backslash is redundant between brackets",
    "E702": "multiple statements on one line (semicolon)",
    "E714": "test for object identity should be 'is not'",
    "E731": "do not assign a lambda expression, use a def",
    "E114": "indentation is not a multiple of four (comment)",
    "E227": "missing whitespace around bitwise or shift operator",
    "E251": "unexpected spaces around keyword / parameter equals",
    "E242": "tab after ','",
    "E241": "multiple spaces after ','",
    "E501": "line too long (82 > 79 characters)",
    "E133": "closing bracket is missing indentation",
    "E226": "missing whitespace around arithmetic operator",
    "E131": "continuation line unaligned for hanging indent",
    "E711": "comparison to None should be 'if cond is None:'",
    "E124": "closing bracket does not match visual indentation",
    "E127": "continuation line over-indented for visual indent",
    "E122": "continuation line missing indentation or outdented",
    "E126": "continuation line over-indented for hanging indent",
    "E128": "continuation line under-indented for visual indent",
    "E121": "continuation line under-indented for hanging indent",
    "E125": "continuation line with same indent as next logical line",
    "E129": "visually indented line with same indent as next logical line",
    "E712": "comparison to True should be 'if cond is True:' or 'if cond:'",
    "E123": "closing bracket does not match indentation of opening bracket's line",

    # McCabe
    'C901': "function is too complex",

    # Naming",
    "N801": "class names should use CapWords convention",
    "N802": "function name should be lowercase",
    "N803": "argument name should be lowercase",
    "N804": "first argument of a classmethod should be named 'cls'",
    "N805": "first argument of a method should be named 'self'",
    "N806": "variable in function should be lowercase",
    "N811": "constant imported as non constant",
    "N812": "lowercase imported as non lowercase",
    "N813": "camelcase imported as lowercase",
    "N814": "camelcase imported as constant",

    # PyFlakes
    "F401": "module imported but unused",
    "F402": "import module from line N shadowed by loop variable",
    "F403": "'from module import *' used; unable to detect undefined names",
    "F404": "future import(s) name after other statements",
    "F811": "redefinition of unused name from line N",
    "F812": "list comprehension redefines name from line N",
    "F821": "undefined name name",
    "F822": "undefined name name in __all__",
    "F823": "local variable name ... referenced before assignment",
    "F831": "duplicate argument name in function definition",
    "F841": "local variable name is assigned to but never used",
}


class CodeTree(dict):
    """
    >>> x = CodeTree(('mycode/module.py', 10, 2, 'E225', 'missing whitespace'))
    >>> x = CodeTree(('mycode/module.py', 10, 2, 'E225', 'missing whitespace'))
    """
    pass


def main(filename):
    """
    Runs the program
    """
    filedata = None
    parsing_error_count = 0
    parsed_rows = []

    with open(filename, 'r') as f:
        filedata = f.readlines()

    if filedata is None:
        print("No file data")
        exit(1)

    for filerow in filedata:
        try:
            code, error, msg = filerow.split(' ', 2)
        except ValueError:
            parsing_error_count += 1
            continue
        try:
            module, line, column = code.split(':')[:3]
        except ValueError:
            parsing_error_count += 1
            continue
        parsed_rows.append((module, line, column, error, msg))

    print("Skipped {} rows due to parsing errors".format(parsing_error_count))

    module_stats = pep8_module_stats(parsed_rows)
    format_stats(module_stats)
    format_error_summary(module_stats)


def pep8_module_stats(parsed_rows):
    """
    :param parsed_rows: list of tuples
    :returns: dictionary like {'module/file.py': {'E225': 5, 'E110': 1}, ... }
    """
    error_stats = {}
    for row in parsed_rows:
        file_name, err = row[0], row[3]
        if file_name not in error_stats:
            error_stats[file_name] = {}
        if err in error_stats[file_name]:
            error_stats[file_name][err] += 1
        else:
            error_stats[file_name][err] = 1
    return error_stats


def format_stats(stats):
    """
    Prints an error stats dictionary of the sum of each error across all files.
    """
    ordered_keys = sorted(stats.keys())
    for key in ordered_keys:
        print(key)
        print("~" * len(key))

        # Sorted list of counts (values) as keys
        counted_stats = OrderedDict(sorted(stats[key].items(),
            key=lambda t: t[1], reverse=True))
        for stat in counted_stats:
            print("({}) {}: {}".format(counted_stats[stat], stat,
                ERROR_CODES.get(stat, "<code not matched>")))
        print("\n")


def format_error_summary(stats):
    """
    Prints a summary of errors across all modules by major category
    """
    group_counts = {key: 0 for key in ERROR_CODE_GROUPS.keys()}
    key_errors = []
    for module, errors in stats.items():
        for error, count in errors.items():
            try:
                group_counts[error[:2]] += count
            except KeyError:
                key_errors.append(error)
                continue

    ordered_errors = OrderedDict(sorted(group_counts.items(),
        key=lambda t: t[1], reverse=True))

    header = "Errors & warnings summary"
    print(header)
    print("-" * len(header))

    for error, total in ordered_errors.items():
        print("{desc} ({err}): {count}".format(
            desc=ERROR_CODE_GROUPS.get(error, "Unknown"),
            err=error, count=total))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("You must provide 1 filename argument")
        exit(1)

    main(sys.argv[1])