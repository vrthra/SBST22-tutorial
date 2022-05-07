# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Prerequisites

import sys

major, minor,*_ = sys.version_info
assert (major,minor) == (3, 10)

# !rm -rf build src subjects

# !mkdir -p build src subjects

# + [markdown] heading_collapsed=true
# ## Python Packages

# + hidden=true
# %pip install graphviz==0.19.2 fuzzingbook==1.0.7 sympy==1.10.1 z3-solver==4.8.16.0 ipynb==0.5.1

# + [markdown] hidden=true
# **IMPORTANT:** Restart the jupyter kernal after installation of dependencies and extensions.

# + [markdown] hidden=true
# ### Jupyter Extensions

# + [markdown] hidden=true
# We recommend the following jupyter notebook extensions:

# + hidden=true
# %pip install jupyter_contrib_nbextensions jupyter_nbextensions_configurator

# + hidden=true
# !{sys.executable} -m jupyter contrib nbextension install --sys-prefix

# + hidden=true
# !{sys.executable} -m jupyter nbextensions_configurator enable

# + [markdown] hidden=true
# #### Table of contents
#
# Please install this extension. The navigation in the notebook is rather hard without this installed.

# + hidden=true
# !{sys.executable} -m jupyter nbextension enable toc2/main

# + [markdown] hidden=true
# #### Collapsible headings
#
# Again, do install this extension. This will let you fold away those sections that you do not have an immediate interest in.

# + hidden=true
# !{sys.executable} -m jupyter nbextension enable collapsible_headings/main

# + [markdown] hidden=true
# #### Code folding
# Very helpful for hiding away source contents of libraries that are not for grammar recovery.

# + hidden=true
# !{sys.executable} -m jupyter nbextension enable codefolding/main
# -

# ## Utils.py

# +
# %%writefile src/utils.py
import sys
import subprocess
from subprocess import run
import os
import json
CMD_TIMEOUT=60*60*24

class O:
    def __init__(self, **keys): self.__dict__.update(keys)
    def __repr__(self): return str(self.__dict__)

def do(command, env=None, shell=False, log=False, inputv=None, timeout=CMD_TIMEOUT, **args):
    result = None
    if inputv:
        result = subprocess.Popen(command,
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            shell = shell,
            env=dict(os.environ, **({} if env is None else env))
        )
        result.stdin.write(inputv)
        stdout, stderr = result.communicate(timeout=timeout)
    else:
        result = subprocess.Popen(command,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            shell = shell,
            env=dict(os.environ, **({} if env is None else env))
        )
        stdout, stderr = result.communicate(timeout=timeout)
    if log:
         with open('build/do.log', 'a+') as f:
            print(json.dumps({'cmd':command,
                              'env':env,
                              'exitcode':result.returncode}), env,
                  flush=True, file=f)
    stdout = '' if stdout is None else stdout.decode()
    stderr = '' if stderr is None else stderr.decode()
    result.kill()
    return O(returncode=result.returncode, stdout=stdout, stderr=stderr)


# +
# %%writefile -a src/utils.py
import types

def slurp(fn):
    with open(fn) as f: return f.read()

def load_src(src, mn):
    module = types.ModuleType(mn)
    exec(src, module.__dict__)
    return module

def load_file(fn, mn):
    return load_src(slurp(fn), mn)


# -

# %%writefile -a src/utils.py
class ExpectError:
    def __init__(self, log=True):
        self.msg = None
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None: return
        self.msg = str(exc_value)
        if self.log:
            print(self.msg, file=sys.stderr)
        return True


# ### Deep Copy

# Python deepcopy is recursive; so it can't do the kind of recursive datastructures we use.

# +
# %%writefile -a src/utils.py
def to_tlv(ds):
    expanded = []
    to_expand = [ds]
    while to_expand:
        ds, *to_expand = to_expand
        if type(ds) in {list, set, tuple}:
            expanded.append(type(ds))
            expanded.append(len(ds))
            to_expand = list(ds) + to_expand
        elif type(ds) in {dict}:
            expanded.append(type(ds))
            expanded.append(len(ds))
            to_expand = list(ds.items()) + to_expand
        else:
            expanded.append(ds)
    return list(reversed(expanded))

def from_tlv(stk):
    def get_children(result_stk):
        l = result_stk.pop()
        return [result_stk.pop() for i in range(l)]
    i = 0
    result_stk = []
    while stk:
        item, *stk = stk
        if item == list:
            ds = get_children(result_stk)
            result_stk.append(ds)
        elif item == set:
            ds = get_children(result_stk)
            result_stk.append(set(ds))
        elif item == tuple:
            ds = get_children(result_stk)
            result_stk.append(tuple(ds))
        elif item == dict:
            ds = get_children(result_stk)
            result_stk.append({i[0]:i[1]for i in ds})
        else:
            result_stk.append(item)
    return result_stk[0]


def deep_copy(arr):
    val = to_tlv(arr)
    return from_tlv(val)


# -

# ### Display

# +
# %%writefile -a src/utils.py
OPTIONS = O(V='|', H='-', L='+', J = '+')

def format_node(node):
    key = node[0]
    if key and (key[0], key[-1]) ==  ('<', '>'): return key
    return repr(key)

def get_children(node):
    return node[1]

def display_tree_console(node, format_node=format_node, get_children=get_children,
                 options=OPTIONS):
    print(format_node(node))
    for line in format_tree(node, format_node, get_children, options):
        print(line)
        
def format_tree(node, format_node, get_children, options, prefix=''):
    children = get_children(node)
    if not children: return
    *children, last_child = children
    for child in children:
        next_prefix = prefix + options.V + '   '
        yield from format_child(child, next_prefix, format_node, get_children,
                                options, prefix, False)
    last_prefix = prefix + '    '
    yield from format_child(last_child, last_prefix, format_node, get_children,
                            options, prefix, True)

def format_child(child, next_prefix, format_node, get_children, options,
                 prefix, last):
    sep = (options.L if last else options.J)
    yield prefix + sep + options.H + ' ' + format_node(child)
    yield from format_tree(child, format_node, get_children, options, next_prefix)


# +
# %%writefile -a src/utils.py
from graphviz import Digraph    
from IPython.display import display                                                                            
import re

class DisplayTree:
    def __init__(self, derivation_tree, verbose=False):
        self.derivation_tree = derivation_tree
        self.counter = 0
        self.verbose = verbose

    def unicode_escape(self, s, error = 'backslashreplace'):              
        def ascii_chr(byte):                                             
            if 0 <= byte <= 127:                                                     
                return chr(byte)                                                     
            return r"\x%02x" % byte                                                  

        bytes = s.encode('utf-8', error)                                             
        return "".join(map(ascii_chr, bytes))

    def dot_escape(self, s):                                                   
        """Return s in a form suitable for dot"""                                    
        s = re.sub(r'([^a-zA-Z0-9" ])', r"\\\1", s)                                  
        return s                                                                     

    def extract_node(self, node, id):                                                      
        symbol, children, *annotation = node                                         
        return symbol, children, ''.join(str(a) for a in annotation)                 

    def node_attr(self, dot, nid, symbol, ann):                                    
        dot.node(repr(nid), self.dot_escape(self.unicode_escape(symbol)))                      

    def edge_attr(self, dot, start_node, stop_node):                               
        dot.edge(repr(start_node), repr(stop_node))                                  

    def graph_attr(self, dot):                                                     
        dot.attr('node', shape='plain')                                              

    def traverse_tree(self, dot, tree, id=0):                                          
        (symbol, children, annotation) = self.extract_node(tree, id)                  
        self.node_attr(dot, id, symbol, annotation)                                   

        if children:                                                             
            for child in children:                                               
                self.counter += 1                                                     
                child_id = self.counter                                               
                self.edge_attr(dot, id, child_id)                                     
                self.traverse_tree(dot, child, child_id)

    def display(self):                                                            
        dot = Digraph(comment="Derivation Tree")                                     
        self.graph_attr(dot)                                                              
        self.traverse_tree(dot, self.derivation_tree)                                          
        if self.verbose:                                                                
            print(dot)                                                               
        return dot

def display_tree(tree, verbose=0):
    return DisplayTree(tree, verbose).display()


# +
# %%writefile -a src/utils.py
def sort_grammar(grammar, start_symbol):
    order = [start_symbol]
    undefined = recurse_grammar(grammar, start_symbol, order)
    return order, [k for k in grammar if k not in order], undefined

def recurse_grammar(grammar, key, order, undefined=None):
    undefined = undefined or {}
    rules = sorted(grammar[key])
    old_len = len(order)
    for rule in rules:
        for token in rule:
            if not is_nt(token): continue
            if token not in grammar:
                if token in undefined:
                    undefined[token].append(key)
                else:
                    undefined[token] = [key]
                continue
            if token not in order:
                order.append(token)
    new = order[old_len:]
    for ckey in new:
        recurse_grammar(grammar, ckey, order, undefined)
    return undefined

class DisplayGrammar:
    def __init__(self, grammar, verbose=0):
        self.grammar = grammar
        self.verbose = verbose

    def is_nonterminal(self, key):
        return is_nt(key)

    def display_token(self, t):
        return t if self.is_nonterminal(t) else repr(t)

    def display_rule(self, rule, pre):
        if self.verbose > -2:
            v = (' '.join([self.display_token(t) for t in rule]))
            s = '%s|   %s' % (pre, v)
            print(s)

    def display_definition(self, key, rule_count):
        if self.verbose > -2: print(key,'::=')
        for rule in self.grammar[key]:
            rule_count += 1
            if self.verbose > 1:
                pre = rule_count
            else:
                pre = ''
            self.display_rule(rule, pre)
        return rule_count

    def display_unused(self, not_used, r):
        if not_used and self.verbose > -1:
            print('[not_used]')
            for key in not_used:
                r = self.display_definition(key, r)
                if self.verbose > 0:
                    print(k, r)

    def display_undefined(self, undefined):
        if undefined and self.verbose > -1:
            print('[undefined keys]')
            for key in undefined:
                if self.verbose == 0:
                    print(key)
                else:
                    print(key, 'defined in')
                    for k in undefined[key]: print(' ', k)

    def display_summary(self, k, r):
        if self.verbose > -1:
            print('keys:', k, 'rules:', r)

    def display(self, start):
        rule_count, key_count = 0, 0
        order, not_used, undefined = sort_grammar(self.grammar, start)
        print('[start]:', start)
        for key in order:
            key_count += 1
            rule_count = self.display_definition(key, rule_count)
            if self.verbose > 0:
                print(key_count, rule_count)

        self.display_unused(not_used, rule_count)
        self.display_undefined(undefined)
        self.display_summary(key_count, rule_count)

def display_grammar(grammar, start, verbose=0):
    DisplayGrammar(grammar, verbose).display(start)


# +
# %%writefile -a src/utils.py
def tree_to_str(tree):
    expanded = []
    to_expand = [tree]
    while to_expand:
        (key, children, *rest), *to_expand = to_expand
        if is_nt(key):
            to_expand = list(children) + list(to_expand)
        else:
            assert not children
            expanded.append(key)
    return ''.join(expanded)

def is_nt(v):
    return v and (v[0], v[-1]) == ('<', '>')

# A token is a lexer token from ANTLR. It is all uppercase nonterminal
# but defined as a regular expression. For example <DIGITS>
def is_token(val):
    assert val != '<>'
    assert (val[0], val[-1]) == ('<', '>')
    if val[1].isupper(): return True
    #if val[1] == '_': return val[2].isupper() # token derived.
    return False


# +
# %%writefile -a src/utils.py
# Grammar Cleanup
def copy_grammar(g):
    return {k:[[t for t in r] for r in g[k]] for k in g}

def find_empty_keys(g):
    return [k for k in g if not g[k]]

def remove_nonterminal(nt, g):
    new_g = {}
    for k_ in g:
        if k_ == nt: continue
        new_rules = []
        for rule in g[k_]:
            if any(t == nt for t in rule): continue
            new_rules.append(rule)
        new_g[k_] = new_rules
    return new_g

def remove_empty_nonterminals(g):
    new_g = copy_grammar(g)
    removed_keys = []
    empty_keys = find_empty_keys(new_g)
    while empty_keys:
        for k in empty_keys:
            removed_keys.append(k)
            new_g = remove_nonterminal(k, new_g)
        empty_keys = find_empty_keys(new_g)
    return new_g, removed_keys

def grammar_gc(grammar, start, remove_unreachable=False):
    new_grammar, removed = remove_empty_nonterminals(grammar)
    if remove_unreachable:
        order, not_used, undefined = sort_grammar(grammar, start)
        return {k:new_grammar[k] for k in order}, start
    return new_grammar, start



# -

# ## Check.py

# +
# %%writefile build/check.py
import sys, imp
parse_ = imp.new_module('parse_')

def init_module(src):
    with open(src) as sf:
        exec(sf.read(), parse_.__dict__)

def _check(s):
    try:
        parse_.main(s)
        return True
    except:
        return False

import sys
def main(args):
    init_module(args[0])
    if _check(args[1]):
        sys.exit(0)
    else:
        sys.exit(1)
import sys
main(sys.argv[1:])

# -

# ## Subject Programs

program_src = {}

# ### Calculator.py

# +
# %%writefile subjects/calculator.py
import string
class MyException(Exception):
    def __init__(self, s, i):
        self.s = s
        self.i = i

def is_digit(i):
    return i in string.digits
    
def parse_num(s,i):
    n = ''
    while s[i:] and is_digit(s[i]):
        n += s[i]
        i = i +1
    return i,n

def parse_paren(s, i):
    assert s[i] == '('
    i, v = parse_expr(s, i+1)
    if s[i:] == '':
        raise Exception(s, i)
    assert s[i] == ')'
    return i+1, v

def parse_expr(s, i = 0):
    expr = []
    is_op = True
    while s[i:]:
        c = s[i]
        if c in string.digits:
            if not is_op: raise Exception(s,i)
            i,num = parse_num(s,i)
            expr.append(num)
            is_op = False
        elif c in ['+', '-', '*', '/']:
            if is_op: raise Exception(s,i)
            expr.append(c)
            is_op = True
            i = i + 1
        elif c == '(':
            if not is_op: raise Exception(s,i)
            i, cexpr = parse_paren(s, i)
            expr.append(cexpr)
            is_op = False
        elif c == ')':
            break
        else:
            raise Exception(s,i)
    if is_op:
        raise Exception(s,i)
    return i, expr

def main(arg):
    i, s = parse_expr(arg)
    if len(arg) != i:
        raise Exception(arg, i)
    return i, s


# -

# ### Microjson.py

# +
# %%writefile subjects/myio.py
r"""File-like objects that read from or write to a string buffer.

This implements (nearly) all stdio methods.

f = StringIO()      # ready for writing
f = StringIO(buf)   # ready for reading
f.close()           # explicitly release resources held
flag = f.isatty()   # always false
pos = f.tell()      # get current position
f.seek(pos)         # set current position
f.seek(pos, mode)   # mode 0: absolute; 1: relative; 2: relative to EOF
buf = f.read()      # read until EOF
buf = f.read(n)     # read up to n bytes
buf = f.readline()  # read until end of line ('\n') or EOF
list = f.readlines()# list of f.readline() results until EOF
f.truncate([size])  # truncate file at to at most size (default: current pos)
f.write(buf)        # write at current position
f.writelines(list)  # for line in list: f.write(line)
f.getvalue()        # return whole file's contents as a string

Notes:
- Using a real file is often faster (but less convenient).
- There's also a much faster implementation in C, called cStringIO, but
  it's not subclassable.
- fileno() is left unimplemented so that code which uses it triggers
  an exception early.
- Seeking far beyond EOF and then writing will insert real null
  bytes that occupy space in the buffer.
- There's a simple test set (see end of this file).
"""
try:
    from errno import EINVAL
except ImportError:
    EINVAL = 22

__all__ = ["StringIO"]

def _complain_ifclosed(closed):
    if closed:
        raise ValueError("I/O operation on closed file")

class StringIO:
    """class StringIO([buffer])

    When a StringIO object is created, it can be initialized to an existing
    string by passing the string to the constructor. If no string is given,
    the StringIO will start empty.

    The StringIO object can accept either Unicode or 8-bit strings, but
    mixing the two may take some care. If both are used, 8-bit strings that
    cannot be interpreted as 7-bit ASCII (that use the 8th bit) will cause
    a UnicodeError to be raised when getvalue() is called.
    """
    def __init__(self, buf = ''):
        # Force self.buf to be a string or unicode
        if not isinstance(buf, str):
            buf = str(buf)
        self.buf = buf
        self.len = len(buf)
        self.buflist = []
        self.pos = 0
        self.closed = False
        self.softspace = 0

    def __iter__(self):
        return self

    def __next__(self):
        """A file object is its own iterator, for example iter(f) returns f
        (unless f is closed). When a file is used as an iterator, typically
        in a for loop (for example, for line in f: print line), the next()
        method is called repeatedly. This method returns the next input line,
        or raises StopIteration when EOF is hit.
        """
        _complain_ifclosed(self.closed)
        r = self.readline()
        if not r:
            raise StopIteration
        return r

    def close(self):
        """Free the memory buffer.
        """
        if not self.closed:
            self.closed = True
            del self.buf, self.pos

    def isatty(self):
        """Returns False because StringIO objects are not connected to a
        tty-like device.
        """
        _complain_ifclosed(self.closed)
        return False

    def seek(self, pos, mode = 0):
        """Set the file's current position.

        The mode argument is optional and defaults to 0 (absolute file
        positioning); other values are 1 (seek relative to the current
        position) and 2 (seek relative to the file's end).

        There is no return value.
        """
        _complain_ifclosed(self.closed)
        if self.buflist:
            self.buf += ''.join(self.buflist)
            self.buflist = []
        if mode == 1:
            pos += self.pos
        elif mode == 2:
            pos += self.len
        self.pos = max(0, pos)

    def tell(self):
        """Return the file's current position."""
        _complain_ifclosed(self.closed)
        return self.pos

    def read(self, n = -1):
        """Read at most size bytes from the file
        (less if the read hits EOF before obtaining size bytes).

        If the size argument is negative or omitted, read all data until EOF
        is reached. The bytes are returned as a string object. An empty
        string is returned when EOF is encountered immediately.
        """
        _complain_ifclosed(self.closed)
        if self.buflist:
            self.buf += ''.join(self.buflist)
            self.buflist = []
        if n is None or n < 0:
            newpos = self.len
        else:
            newpos = min(self.pos+n, self.len)
        r = self.buf[self.pos:newpos]
        self.pos = newpos
        return r

    def readline(self, length=None):
        r"""Read one entire line from the file.

        A trailing newline character is kept in the string (but may be absent
        when a file ends with an incomplete line). If the size argument is
        present and non-negative, it is a maximum byte count (including the
        trailing newline) and an incomplete line may be returned.

        An empty string is returned only when EOF is encountered immediately.

        Note: Unlike stdio's fgets(), the returned string contains null
        characters ('\0') if they occurred in the input.
        """
        _complain_ifclosed(self.closed)
        if self.buflist:
            self.buf += ''.join(self.buflist)
            self.buflist = []
        i = self.buf.find('\n', self.pos)
        if i < 0:
            newpos = self.len
        else:
            newpos = i+1
        if length is not None and length > 0:
            if self.pos + length < newpos:
                newpos = self.pos + length
        r = self.buf[self.pos:newpos]
        self.pos = newpos
        return r

    def readlines(self, sizehint = 0):
        """Read until EOF using readline() and return a list containing the
        lines thus read.

        If the optional sizehint argument is present, instead of reading up
        to EOF, whole lines totalling approximately sizehint bytes (or more
        to accommodate a final whole line).
        """
        total = 0
        lines = []
        line = self.readline()
        while line:
            lines.append(line)
            total += len(line)
            if 0 < sizehint <= total:
                break
            line = self.readline()
        return lines

    def truncate(self, size=None):
        """Truncate the file's size.

        If the optional size argument is present, the file is truncated to
        (at most) that size. The size defaults to the current position.
        The current file position is not changed unless the position
        is beyond the new file size.

        If the specified size exceeds the file's current size, the
        file remains unchanged.
        """
        _complain_ifclosed(self.closed)
        if size is None:
            size = self.pos
        elif size < 0:
            raise IOError(EINVAL, "Negative size not allowed")
        elif size < self.pos:
            self.pos = size
        self.buf = self.getvalue()[:size]
        self.len = size

    def write(self, s):
        """Write a string to the file.

        There is no return value.
        """
        _complain_ifclosed(self.closed)
        if not s: return
        # Force s to be a string or unicode
        if not isinstance(s, str):
            s = str(s)
        spos = self.pos
        slen = self.len
        if spos == slen:
            self.buflist.append(s)
            self.len = self.pos = spos + len(s)
            return
        if spos > slen:
            self.buflist.append('\0'*(spos - slen))
            slen = spos
        newpos = spos + len(s)
        if spos < slen:
            if self.buflist:
                self.buf += ''.join(self.buflist)
            self.buflist = [self.buf[:spos], s, self.buf[newpos:]]
            self.buf = ''
            if newpos > slen:
                slen = newpos
        else:
            self.buflist.append(s)
            slen = newpos
        self.len = slen
        self.pos = newpos

    def writelines(self, iterable):
        """Write a sequence of strings to the file. The sequence can be any
        iterable object producing strings, typically a list of strings. There
        is no return value.

        (The name is intended to match readlines(); writelines() does not add
        line separators.)
        """
        write = self.write
        for line in iterable:
            write(line)

    def flush(self):
        """Flush the internal buffer
        """
        _complain_ifclosed(self.closed)

    def getvalue(self):
        """
        Retrieve the entire contents of the "file" at any time before
        the StringIO object's close() method is called.

        The StringIO object can accept either Unicode or 8-bit strings,
        but mixing the two may take some care. If both are used, 8-bit
        strings that cannot be interpreted as 7-bit ASCII (that use the
        8th bit) will cause a UnicodeError to be raised when getvalue()
        is called.
        """
        if self.buflist:
            self.buf += ''.join(self.buflist)
            self.buflist = []
        return self.buf


# +
# %%writefile subjects/microjson.py
# microjson - Minimal JSON parser/emitter for use in standalone scripts.
# No warranty. Free to use/modify as you see fit. Trades speed for compactness.
# Send ideas, bugs, simplifications to http://github.com/phensley
# Copyright (c) 2010 Patrick Hensley <spaceboy@indirect.com>

# std
import math
import subjects.myio as io
import types


# the '_from_json_number' function returns either float or long.
__pychecker__ = 'no-returnvalues'

# character classes
WS = ' ' # ''.join([' ','\t','\r','\n','\b','\f'])
DIGITS = ''.join([str(i) for i in range(0, 10)])
NUMSTART = DIGITS + ''.join(['.','-','+'])
NUMCHARS = NUMSTART + ''.join(['e','E'])
ESC_MAP = {'n':'\n','t':'\t','r':'\r','b':'\b','f':'\f'}
REV_ESC_MAP = dict([(_v,_k) for _k,_v in list(ESC_MAP.items())] + [('"','"')])

# error messages
E_BYTES = 'input string must be type str containing ASCII or UTF-8 bytes'
E_MALF = 'malformed JSON data'
E_TRUNC = 'truncated JSON data'
E_BOOL = 'expected boolean'
E_NULL = 'expected null'
E_LITEM = 'expected list item'
E_DKEY = 'expected key'
E_COMMA = 'missing comma between elements'
E_COLON = 'missing colon after key'
E_EMPTY = 'found empty string, not valid JSON data'
E_BADESC = 'bad escape character found'
E_UNSUPP = 'unsupported type "%s" cannot be JSON-encoded'
E_BADFLOAT = 'cannot emit floating point value "%s"'
E_EXTRA = 'extra data after JSON'

NEG_INF = float('-inf')
POS_INF = float('inf')


class JSONError(Exception):
    def __init__(self, msg, stm=None, pos=0):
        if stm:
            msg += ' at position %d, "%s"' % (pos, repr(stm.substr(pos, 32)))
        Exception.__init__(self, msg)
        self.pos = pos


class JSONStream:

    # no longer inherit directly from StringIO, since we only want to
    # expose the methods below and not allow direct access to the
    # underlying stream.

    def __init__(self, data):
        self._stm = io.StringIO(data)

    @property
    def pos(self):
        return self._stm.tell()

    @property
    def len(self):
        return len(self._stm.getvalue())

    def getvalue(self):
        return self._stm.getvalue()

    def skipspaces(self):
        "post-cond: read pointer will be over first non-WS char"
        self._skip(lambda c: not c in WS)

    def _skip(self, stopcond):
        while True:
            c = self.peek()
            if stopcond(c) or c == '':
                break
            self.next()

    def next(self, size=1):
        return self._stm.read(size)

    def next_ord(self):
        return ord(next(self))

    def peek(self):
        if self.pos == self.len:
            return self.getvalue()[self.pos:]
        return self.getvalue()[self.pos]

    def substr(self, pos, length):
        return self.getvalue()[pos:pos+length]


def _decode_utf8(c0, stm):
    c0 = ord(c0)
    r = 0xFFFD      # unicode replacement character
    nc = stm.next_ord

    # 110yyyyy 10zzzzzz
    if (c0 & 0xE0) == 0xC0:
        r = ((c0 & 0x1F) << 6) + (nc() & 0x3F)

    # 1110xxxx 10yyyyyy 10zzzzzz
    elif (c0 & 0xF0) == 0xE0:
        r = ((c0 & 0x0F) << 12) + ((nc() & 0x3F) << 6) + (nc() & 0x3F)

    # 11110www 10xxxxxx 10yyyyyy 10zzzzzz
    elif (c0 & 0xF8) == 0xF0:
        r = ((c0 & 0x07) << 18) + ((nc() & 0x3F) << 12) + \
            ((nc() & 0x3F) << 6) + (nc() & 0x3F)
    return chr(r)


def decode_escape(c, stm):
    # whitespace
    v = ESC_MAP.get(c, None)
    if v is not None:
        return v

    # plain character
    elif c != 'u':
        return c

    # decode unicode escape \u1234
    sv = 12
    r = 0
    for _ in range(0, 4):
        r |= int(stm.next(), 16) << sv
        sv -= 4
    return chr(r)


def _from_json_string(stm):
    try:
        # skip over '"'
        stm.next()
        r = ''
        while True:
            c = stm.next()
            if c == '':
                raise JSONError(E_TRUNC, stm, stm.pos - 1)
            elif c == '\\':
                c = stm.next()
                r += decode_escape(c, stm)
            elif c == '"':
                return r
            elif c in [str(i) for i in range(127, 256)]:
                r += _decode_utf8(c, stm)
            else:
                r += c
    except ValueError as v:
        raise JSONError(E_MALF, stm, stm.pos)
        

def _from_json_fixed(stm, expected, value, errmsg):
    off = len(expected)
    pos = stm.pos
    res = stm.substr(pos, off)
    if res == expected:
        stm.next(off)
        return res
    raise JSONError(errmsg, stm, pos)


def _from_json_number(stm):
    # Per rfc 4627 section 2.4 '0' and '0.1' are valid, but '01' and
    # '01.1' are not, presumably since this would be confused with an
    # octal number.  This rule is not enforced.
    is_float = 0
    saw_exp = 0
    pos = stm.pos
    while True:
        c = stm.peek()
        if not c: break

        if not c in NUMCHARS:
            break
        elif c == '-' and not saw_exp:
            pass
        elif c in '.eE':
            is_float = 1
            if c in 'eE':
                saw_exp = 1

        stm.next()

    s = stm.substr(pos, stm.pos - pos)
    if is_float:
        return s
    return s


def _from_json_list(stm):
    # skip over '['
    stm.next()
    result = []
    pos = stm.pos
    comma = False
    while True:
        stm.skipspaces()
        c = stm.peek()
        if c == '':
            raise JSONError(E_TRUNC, stm, pos)

        elif c == ']':
            stm.next()
            return result

        elif c == ',':
            if not result:
                raise JSONError(E_TRUNC, stm, pos)
            if comma:
                raise JSONError(E_TRUNC, stm, pos)
            comma = True
            stm.next()
            result.append(_from_json_raw(stm))
            comma = False
            continue

        elif not result:
            # first item
            result.append(_from_json_raw(stm))
            comma = False
            continue

        else:
            raise JSONError(E_MALF, stm, stm.pos)


def _from_json_dict(stm):
    # skip over '{'
    stm.next()
    result = {}
    expect_key = 1
    pos = stm.pos
    comma = False
    while True:
        stm.skipspaces()
        c = stm.peek()
        if c == '':
            raise JSONError(E_TRUNC, stm, pos)

        # end of dictionary, or next item
        elif c == '}':
            if expect_key == 2:
                raise JSONError(E_TRUNC, stm, pos)
            stm.next()
            return result

        elif c == ',':
            if not result:
                raise JSONError(E_TRUNC, stm, pos)
            if comma:
                raise JSONError(E_TRUNC, stm, pos)
            comma = True
            stm.next()
            expect_key = 2
            continue

        # parse out a key/value pair
        elif c == '"':
            if not expect_key:
                raise JSONError(E_COMMA, stm, stm.pos)
            key = _from_json_string(stm)
            stm.skipspaces()
            c = stm.next()
            if c != ':':
                raise JSONError(E_COLON, stm, stm.pos)

            stm.skipspaces()
            val = _from_json_raw(stm)
            result[key] = val
            expect_key = 0
            comma = False
            continue

        # unexpected character in middle of dict
        raise JSONError(E_MALF, stm, stm.pos)


def _from_json_raw(stm):
    while True:
        stm.skipspaces()
        c = stm.peek()
        if c == '"': 
            return _from_json_string(stm)
        elif c == '{': 
            return _from_json_dict(stm)
        elif c == '[': 
            return _from_json_list(stm)
        elif c == 't':
            return _from_json_fixed(stm, 'true', True, E_BOOL)
        elif c == 'f':
            return _from_json_fixed(stm, 'false', False, E_BOOL)
        elif c == 'n': 
            return _from_json_fixed(stm, 'null', None, E_NULL)
        elif c in NUMSTART:
            return _from_json_number(stm)

        raise JSONError(E_MALF, stm, stm.pos)


def from_json(data):
    """
    Converts 'data' which is UTF-8 (or the 7-bit pure ASCII subset) into
    a Python representation.  You must pass bytes to this in a str type,
    not unicode.
    """
    if not isinstance(data, str):
        raise JSONError(E_BYTES)
    if not data:
        return None
    stm = JSONStream(data)
    v = _from_json_raw(stm)
    c = stm.peek()
    if c:
        raise JSONError(E_EXTRA, stm, stm.pos)
    return v


# JSON emitter

def _to_json_list(stm, lst):
    seen = 0
    stm.write('[')
    for elem in lst:
        if seen:
            stm.write(',')
        seen = 1
        _to_json_object(stm, elem)
    stm.write(']')


def _to_json_string(stm, buf):
    stm.write('"')
    for c in buf:
        nc = REV_ESC_MAP.get(c, None)
        if nc:
            stm.write('\\' + nc)
        elif ord(c) <= 0x7F:
            # force ascii
            stm.write(str(c))
        else:
            stm.write('\\u%04x' % ord(c))
    stm.write('"')


def _to_json_dict(stm, dct):
    seen = 0
    stm.write('{')
    for key in list(dct.keys()):
        if seen:
            stm.write(',')
        seen = 1
        val = dct[key]
        if not type(key) in (bytes, str):
            key = str(key)
        _to_json_string(stm, key)
        stm.write(':')
        _to_json_object(stm, val)
    stm.write('}')


def _to_json_object(stm, obj):
    if isinstance(obj, (list, tuple)):
        _to_json_list(stm, obj)
    elif isinstance(obj, bool):
        if obj:
            stm.write('true')
        else:
            stm.write('false')
    elif isinstance(obj, float):
        # this raises an error for NaN, -inf and inf values
        if not (NEG_INF < obj < POS_INF):
            raise JSONError(E_BADFLOAT % obj)
        stm.write("%s" % obj)
    elif isinstance(obj, int):
        stm.write("%d" % obj)
    elif isinstance(obj, type(None)):
        stm.write('null')
    elif isinstance(obj, (bytes, str)):
        _to_json_string(stm, obj)
    elif hasattr(obj, 'keys') and hasattr(obj, '__getitem__'):
        _to_json_dict(stm, obj)
    # fall back to implicit string conversion.
    elif hasattr(obj, '__unicode__'):
        _to_json_string(stm, obj.__unicode__())
    elif hasattr(obj, '__str__'):
        _to_json_string(stm, obj.__str__())
    else:
        raise JSONError(E_UNSUPP % type(obj))


def to_json(obj):
    """
    Converts 'obj' to an ASCII JSON string representation.
    """
    stm = io.StringIO('')
    _to_json_object(stm, obj)
    return stm.getvalue()


decode = from_json
encode = to_json

def main(arg):
    return from_json(arg)


# +
# %%writefile subjects/mylex.py
"""A lexical analyzer class for simple shell-like syntaxes."""

# Module and documentation by Eric S. Raymond, 21 Dec 1998
# Input stacking and error message cleanup added by ESR, March 2000
# push_source() and pop_source() made explicit by ESR, January 2001.
# Posix compliance, split(), string arguments, and
# iterator interface by Gustavo Niemeyer, April 2003.
# changes to tokenize more like Posix shells by Vinay Sajip, July 2016.

import os
import re
import sys
from collections import deque

from myio import StringIO

__all__ = ["shlex", "split", "quote"]

class shlex:
    "A lexical analyzer class for simple shell-like syntaxes."
    def __init__(self, instream=None, infile=None, posix=False,
                 punctuation_chars=False):
        if isinstance(instream, str):
            instream = StringIO(instream)
        if instream is not None:
            self.instream = instream
            self.infile = infile
        else:
            self.instream = sys.stdin
            self.infile = None
        self.posix = posix
        if posix:
            self.eof = None
        else:
            self.eof = ''
        self.commenters = '#'
        self.wordchars = ('abcdfeghijklmnopqrstuvwxyz'
                          'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
        if self.posix:
            self.wordchars += ('ßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
                               'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ')
        self.whitespace = ' \t\r\n'
        self.whitespace_split = False
        self.quotes = '\'"'
        self.escape = '\\'
        self.escapedquotes = '"'
        self.state = ' '
        self.pushback = deque()
        self.lineno = 1
        self.debug = 0
        self.token = ''
        self.filestack = deque()
        self.source = None
        if not punctuation_chars:
            punctuation_chars = ''
        elif punctuation_chars is True:
            punctuation_chars = '();<>|&'
        self.punctuation_chars = punctuation_chars
        if punctuation_chars:
            # _pushback_chars is a push back queue used by lookahead logic
            self._pushback_chars = deque()
            # these chars added because allowed in file names, args, wildcards
            self.wordchars += '~-./*?='
            #remove any punctuation chars from wordchars
            t = self.wordchars.maketrans(dict.fromkeys(punctuation_chars))
            self.wordchars = self.wordchars.translate(t)

    def push_token(self, tok):
        "Push a token onto the stack popped by the get_token method"
        if self.debug >= 1:
            print("shlex: pushing token " + repr(tok))
        self.pushback.appendleft(tok)

    def push_source(self, newstream, newfile=None):
        "Push an input source onto the lexer's input source stack."
        if isinstance(newstream, str):
            newstream = StringIO(newstream)
        self.filestack.appendleft((self.infile, self.instream, self.lineno))
        self.infile = newfile
        self.instream = newstream
        self.lineno = 1
        if self.debug:
            if newfile is not None:
                print('shlex: pushing to file %s' % (self.infile,))
            else:
                print('shlex: pushing to stream %s' % (self.instream,))

    def pop_source(self):
        "Pop the input source stack."
        self.instream.close()
        (self.infile, self.instream, self.lineno) = self.filestack.popleft()
        if self.debug:
            print('shlex: popping to %s, line %d' \
                  % (self.instream, self.lineno))
        self.state = ' '

    def get_token(self):
        "Get a token from the input stream (or from stack if it's nonempty)"
        if self.pushback:
            tok = self.pushback.popleft()
            if self.debug >= 1:
                print("shlex: popping token " + repr(tok))
            return tok
        # No pushback.  Get a token.
        raw = self.read_token()
        # Handle inclusions
        if self.source is not None:
            while raw == self.source:
                spec = self.sourcehook(self.read_token())
                if spec:
                    (newfile, newstream) = spec
                    self.push_source(newstream, newfile)
                raw = self.get_token()
        # Maybe we got EOF instead?
        while raw == self.eof:
            if not self.filestack:
                return self.eof
            else:
                self.pop_source()
                raw = self.get_token()
        # Neither inclusion nor EOF
        if self.debug >= 1:
            if raw != self.eof:
                print("shlex: token=" + repr(raw))
            else:
                print("shlex: token=EOF")
        return raw

    def read_token(self):
        quoted = False
        escapedstate = ' '
        while True:
            if self.punctuation_chars and self._pushback_chars:
                nextchar = self._pushback_chars.pop()
            else:
                nextchar = self.instream.read(1)
            if nextchar == '\n':
                self.lineno += 1
            if self.debug >= 3:
                print("shlex: in state %r I see character: %r" % (self.state,
                                                                  nextchar))
            if self.state is None:
                self.token = ''        # past end of file
                break
            elif self.state == ' ':
                if not nextchar:
                    self.state = None  # end of file
                    break
                elif nextchar in self.whitespace:
                    if self.debug >= 2:
                        print("shlex: I see whitespace in whitespace state")
                    if self.token or (self.posix and quoted):
                        break   # emit current token
                    else:
                        continue
                elif nextchar in self.commenters:
                    self.instream.readline()
                    self.lineno += 1
                elif self.posix and nextchar in self.escape:
                    escapedstate = 'a'
                    self.state = nextchar
                elif nextchar in self.wordchars:
                    self.token = nextchar
                    self.state = 'a'
                elif nextchar in self.punctuation_chars:
                    self.token = nextchar
                    self.state = 'c'
                elif nextchar in self.quotes:
                    if not self.posix:
                        self.token = nextchar
                    self.state = nextchar
                elif self.whitespace_split:
                    self.token = nextchar
                    self.state = 'a'
                else:
                    self.token = nextchar
                    if self.token or (self.posix and quoted):
                        break   # emit current token
                    else:
                        continue
            elif self.state in self.quotes:
                quoted = True
                if not nextchar:      # end of file
                    if self.debug >= 2:
                        print("shlex: I see EOF in quotes state")
                    # XXX what error should be raised here?
                    raise ValueError("No closing quotation")
                if nextchar == self.state:
                    if not self.posix:
                        self.token += nextchar
                        self.state = ' '
                        break
                    else:
                        self.state = 'a'
                elif (self.posix and nextchar in self.escape and self.state
                      in self.escapedquotes):
                    escapedstate = self.state
                    self.state = nextchar
                else:
                    self.token += nextchar
            elif self.state in self.escape:
                if not nextchar:      # end of file
                    if self.debug >= 2:
                        print("shlex: I see EOF in escape state")
                    # XXX what error should be raised here?
                    raise ValueError("No escaped character")
                # In posix shells, only the quote itself or the escape
                # character may be escaped within quotes.
                if (escapedstate in self.quotes and
                        nextchar != self.state and nextchar != escapedstate):
                    self.token += self.state
                self.token += nextchar
                self.state = escapedstate
            elif self.state in ('a', 'c'):
                if not nextchar:
                    self.state = None   # end of file
                    break
                elif nextchar in self.whitespace:
                    if self.debug >= 2:
                        print("shlex: I see whitespace in word state")
                    self.state = ' '
                    if self.token or (self.posix and quoted):
                        break   # emit current token
                    else:
                        continue
                elif nextchar in self.commenters:
                    self.instream.readline()
                    self.lineno += 1
                    if self.posix:
                        self.state = ' '
                        if self.token or (self.posix and quoted):
                            break   # emit current token
                        else:
                            continue
                elif self.state == 'c':
                    if nextchar in self.punctuation_chars:
                        self.token += nextchar
                    else:
                        if nextchar not in self.whitespace:
                            self._pushback_chars.append(nextchar)
                        self.state = ' '
                        break
                elif self.posix and nextchar in self.quotes:
                    self.state = nextchar
                elif self.posix and nextchar in self.escape:
                    escapedstate = 'a'
                    self.state = nextchar
                elif (nextchar in self.wordchars or nextchar in self.quotes
                      or self.whitespace_split):
                    self.token += nextchar
                else:
                    if self.punctuation_chars:
                        self._pushback_chars.append(nextchar)
                    else:
                        self.pushback.appendleft(nextchar)
                    if self.debug >= 2:
                        print("shlex: I see punctuation in word state")
                    self.state = ' '
                    if self.token or (self.posix and quoted):
                        break   # emit current token
                    else:
                        continue
        result = self.token
        self.token = ''
        if self.posix and not quoted and result == '':
            result = None
        if self.debug > 1:
            if result:
                print("shlex: raw token=" + repr(result))
            else:
                print("shlex: raw token=EOF")
        return result

    def sourcehook(self, newfile):
        "Hook called on a filename to be sourced."
        if newfile[0] == '"':
            newfile = newfile[1:-1]
        # This implements cpp-like semantics for relative-path inclusion.
        if isinstance(self.infile, str) and not os.path.isabs(newfile):
            newfile = os.path.join(os.path.dirname(self.infile), newfile)
        return (newfile, open(newfile, "r"))

    def error_leader(self, infile=None, lineno=None):
        "Emit a C-compiler-like, Emacs-friendly error-message leader."
        if infile is None:
            infile = self.infile
        if lineno is None:
            lineno = self.lineno
        return "\"%s\", line %d: " % (infile, lineno)

    def __iter__(self):
        return self

    def __next__(self):
        token = self.get_token()
        if token == self.eof:
            raise StopIteration
        return token

def split(s, comments=False, posix=True):
    lex = shlex(s, posix=posix)
    lex.whitespace_split = True
    if not comments:
        lex.commenters = ''
    return list(lex)


_find_unsafe = re.compile(r'[^\w@%+=:,./-]', re.ASCII).search

def quote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"
    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _print_tokens(lexer):
    while 1:
        tt = lexer.get_token()
        if not tt:
            break
        print("Token: " + repr(tt))
# -

# # Done

# +
# #%tb
