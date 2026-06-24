# Sprout

A small programming language with a tree-walking interpreter, written from scratch in Python.

## Why

I wanted to actually understand how interpreters work — not just read about lexers and parsers, but build one end to end. So Sprout has no parsing libraries, no dependencies, nothing magic: just a hand-written lexer, a recursive-descent parser, and an interpreter that walks the syntax tree. If you've ever wondered what happens between typing `1 + 2 * 3` and getting `7`, the code here is meant to be read.

## Running it

Run a file:

```sh
python3 main.py examples/fizzbuzz.sprout
```

Or start the REPL (no arguments) and poke at things line by line:

```sh
python3 main.py
>>> let x = 21;
>>> x * 2
42
```

No install step, no packages — you just need Python 3. (The `.sprout` extension is only a convention; any text file works.)

## The language in 60 seconds

```
// variables and the basic types
let name = "Sprout";
let count = 3;
let pi = 3.14;
let ok = true;        // also: false, nil

// arithmetic, comparison, logic
let x = 1 + 2 * 3;    // + - * / %
let big = x > 5 and x < 100;
let either = false or true;

// strings: concatenation and interpolation
print "hi " + name;
print "name={name} count={count} sum={1 + 2}";
print "newlines\tand tabs\nwork too";   // \n \t \r \\ \" \{ \}

// if / else if / else
if (count > 5) { print "lots"; }
else if (count > 0) { print "some"; }
else { print "none"; }

// loops
while (count > 0) { count = count - 1; }
for (let i = 0; i < 5; i = i + 1) { print i; }

// functions are values and can close over variables
fn make_adder(n) {
    fn add(x) { return x + n; }
    return add;
}
let add10 = make_adder(10);
print add10(5);       // 15

// arrays
let nums = [1, 2, 3];
push(nums, 4);
print nums[0];        // 1
print nums[-1];       // 4  (negative indexes count from the end)
nums[0] = 99;

// hash maps
let user = {"name": "ada", "age": 36};
print user["name"];
user["age"] = 37;
print keys(user);     // ["name", "age"]
print has(user, "name");
```

### Built-in functions

| Function | What it does |
|---|---|
| `len(x)` | length of an array, string, or map |
| `push(arr, v)` / `pop(arr)` | append / remove-last on an array |
| `keys(map)` / `has(map, k)` | a map's keys / whether a key exists |
| `range(stop)` / `range(start, stop[, step])` | array of numbers |
| `input(prompt?)` | read a line (returns `nil` at end of input) |
| `str(x)` / `num(s)` | convert to string / parse a number |
| `split(s, sep?)` / `join(arr, sep)` | string ↔ array |

### When something goes wrong

Errors point at the line and the spot:

```
Runtime error: index 10 out of range for length 3 (line 2)

  2 | print a[10];
    | ^
```

## Examples

There are a handful of complete programs in [`examples/`](examples/):

- `fizzbuzz.sprout` — loops and conditionals
- `fibonacci.sprout` — recursion and an iterative version
- `guess.sprout` — a number-guessing game using `input()`
- `wordcount.sprout` — counting words with a hash map
- `stats.sprout` — min/max/sum over an array

## Tests

```sh
python3 -m unittest discover -s tests
```
