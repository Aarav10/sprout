# Sprout

A small programming language I made in Python. It has a tree walking interpreter and I wrote the whole thing from scratch.

## Why

I wanted to actually get how interpreters work instead of just reading about lexers and parsers. So I built one start to finish. No parsing libraries, no dependencies, nothing fancy. Just a lexer I wrote by hand, a recursive descent parser, and an interpreter that walks the tree. If you've ever wondered what actually happens between typing `1 + 2 * 3` and getting `7`, that's basically what this code is.

## Running it

Run a file:

```sh
python3 main.py examples/fizzbuzz.sprout
```

Or just start the REPL with no arguments and mess around line by line:

```sh
python3 main.py
>>> let x = 21;
>>> x * 2
42
```

No install, no packages, you just need Python 3. The `.sprout` extension is just a convention, any text file works.

## The language in 60 seconds

```
// variables and basic types
let name = "Sprout";
let count = 3;
let pi = 3.14;
let ok = true;        // also false, nil

// math, comparison, logic
let x = 1 + 2 * 3;    // + - * / %
let big = x > 5 and x < 100;
let either = false or true;

// strings: concat and interpolation
print "hi " + name;
print "name={name} count={count} sum={1 + 2}";
print "newlines\tand tabs\nwork too";

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
| `push(arr, v)` / `pop(arr)` | add to / remove last from an array |
| `keys(map)` / `has(map, k)` | a map's keys / whether a key exists |
