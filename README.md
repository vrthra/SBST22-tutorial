# SBST22 Tutorial

This is the repository for the [SBST 2022 tutorial](https://sbst22.github.io/keynotes/) on "Learning and Refining Input Grammars for Effective Fuzzing".

One of the concerns in search based software engineering is the search space.
Our algorithms can be more performant if we can constrain this search space.
Hence, the focus of this tutorial is to provide a suite of tools that can be
used in conjunction with search based software engineering to constrain the
search space in testing. That is, we mine the input specifications from
parsers, which are then used to constrain the inputs to be provided to the
program under test. We constrain this space further by specifying that only
inputs belonging to patterns that are known to cause certain behaviors should
be generated.
In particular, we will see:

1. How to generate inputs using [GA](https://en.wikipedia.org/wiki/Genetic_algorithm) when the syntax specification is not available
2. How to use the sample inputs for mining the syntax specification (context-free grammar) of a given parser
3. Given such a specification, how to abstract any input that causes a bug resulting in a bug specification
4. How to combine the specifications of such bugs using *and*, *or* and *negate* for complex bug specifications. For example, one can specify that **each input** produced by a fuzzer should contain input patterns that induce bugs A, B, and either C or D but should not contain bug E.

  That is:
    
    A & B & (C | D) \ E

**Note.** While the algorithms are presented in Python, the techniques presented are language agnostic. That is, you can apply these techniques and the implementation directly to any parser so long as the parser satisfies at least some of these criteria. Furthermore, any criteria that the targeted parser doesn't satisfy can easily be worked around with some (usually minimal) effort.

1. For sections (1) (Generating samples without specification) we assume that the program under test is written in a language that
   1. Allows some way to wrap the input string in a proxy object such that access to the input string can be tracked as long as it remains a string (i.e unparsed).
   2. Failing which (i.e. C), we assume some taint tracking mechanism that allows us to identify the part of the input string we are operating on in any parsing unit.
   3. If no such taint trackers are available, we assume that the parser processes input byte by byte and exits with an error as soon as an unparsable byte is found.

2. For section (2), beyond the requirement from (1) we also assume that the parser
   1. Uses functions/procedures/methods to organize various parsing units
   2. Uses structured programming techniques such as loops and conditionals
   3. Failing these (i.e. combinatorial parsers) the parsing units are labeled (i.e. the variable containing the parsing unit is adequately named), and these labels are recoverable somehow.

3. Sections (3) and (4) does not impose any requirements on the parser.

## Prerequisites

- Download [Python 3.10](https://www.python.org/downloads/)
   
  Some variation of
  ```
  $ sudo apt-get install python3.10
  ```
- Install [Graphviz](https://graphviz.org/download/) for your operating system.
  
  Some variation of
  ```
  $ sudo apt install graphviz
  ```
- Make a virtual environment (recommended)
  ```
  $ python3 -m venv sbst2022
  $ cd sbt2022
  $ source bin/activate
  ```
- Install [Jupyter](https://jupyter.org/).
  ```
  $ ./bin/python3 -m pip install jupyter jupytext
  ```
- Checkout this repository
  ```
  git clone git@github.com:vrthra/SBST22-tutorial.git
  ```
- Start the Jupyter server in the repository directory
  ```
  $ cd SBST22-tutorial
  $ jupyter-notebook
  ```
  This opens a browser window at http://localhost:8888/tree
- Proceed to [x0_0_Prerequisites.ipynb](http://localhost:8888/notebooks/x0_0_Prerequisites.ipynb) and execute the page completely.

- The [Roadmap.ipynb](http://localhost:8888/notebooks/Roadmap.ipynb) contains
  the description of each notebook and links to each.

## Stepping through the tutorial

The jupyter notebooks in this tutorial are designed to be stepped through in an ordered sequence.
For example, The first number `0` in `x0_3_HDD.ipynb` is the major number and `3` is the minor number.
The major number describes the particular section being explained. These are:

0. Prerequisites

   These are well known external algorithms, and will not be explained during the tutorial.
1. Generating Samples

   How to generate valid samples for grammar mining when one does not have access to the input specification for parsers (e.g. JSON, Java syntax etc.).
   
2. Mining Grammar

   How to mine the context-free grammar of a given parser
   
3. Abstracting Inputs

   If you find a bug or an otherwise interesting behavior, how to abstract this input so that you can generate a much larger number of inputs all reproducing this behavior.
   
4. Input Algebras

   If you find multiple such interesting behaviors, or bugs, how to combine them, using the full algebraic operations -- conjunction (and), disjunction (or), and negation (complement) so that each input contains all bugs you specify.

The minor numbers specify the tasks that are involved in each major step.
