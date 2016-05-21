fabfhmm.py: the FAB-FHMM engine. When set doShrink=False, it falls back to Zoubin's VB-FHMM.
fabtest.py: the test program which reads one data file to train, and two files to predict.
artificial.py: generates a test set consisting of 5 data files, using 'param-3d-2,2,3.cfg' (you can also generate with other models by choosing another .cfg or editing the .cfg contents.
groundlike.py: tests how well the ground truth model predicts the test sequences ( it performs badly :( I don't know why )
parselog.pl: parses the log file produced by fabtest.py, and generates statistics.
