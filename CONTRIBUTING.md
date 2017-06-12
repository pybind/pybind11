Thank you for your interest in this project! Please refer to the following
sections on how to contribute code and bug reports.

### Reporting bugs

At the moment, this project is run in the spare time of a single person
([Wenzel Jakob](http://rgl.epfl.ch/people/wjakob)) with very limited resources
for issue tracker tickets. Thus, before submitting a question or bug report,
please take a moment of your time and ensure that your issue isn't already
discussed in the project documentation provided at
[http://pybind11.readthedocs.org/en/latest](http://pybind11.readthedocs.org/en/latest).

Assuming that you have identified a previously unknown problem or an important
question, it's essential that you submit a self-contained and minimal piece of
code that reproduces the problem. In other words: no external dependencies,
isolate the function(s) that cause breakage, submit matched and complete C++
and Python snippets that can be easily compiled and run on my end.

## Pull requests
Contributions are submitted, reviewed, and accepted using Github pull requests.
Please refer to [this
article](https://help.github.com/articles/using-pull-requests) for details and
adhere to the following rules to make the process as smooth as possible:

* Make a new branch for every feature you're working on.
* Make small and clean pull requests that are easy to review but make sure they
  do add value by themselves.
* Add tests for any new functionality and run the test suite (``make pytest``)
  to ensure that no existing features break.
* This project has a strong focus on providing general solutions using a
  minimal amount of code, thus small pull requests are greatly preferred.

### License

pybind11 is provided under a BSD-style license that can be found in the
``LICENSE`` file. By using, distributing, or contributing to this project, you
agree to the terms and conditions of this license.
