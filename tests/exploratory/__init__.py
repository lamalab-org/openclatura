"""Exploratory regression tests for the fix-elided-locant-parentheses branch.

These tests were designed adversarially against the specific string-level
heuristics introduced/refactored in that branch (see the module docstrings
below for the exact functions each file targets), rather than derived from
any single bug report. They exist to positively pin down a broad range of
"is this parenthesis actually necessary" decisions across every central-atom
family the branch touches, plus a couple of pre-existing (not introduced by
this branch) issues uncovered along the way.
"""
