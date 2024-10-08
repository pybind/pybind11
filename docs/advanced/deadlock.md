# Double locking, deadlocking, GIL

[TOC]

## Introduction

### Overview

In concurrent programming with locks, *deadlocks* can arise when more than one
mutex is locked at the same time, and careful attention has to be paid to lock
ordering to avoid this. Here we will look at a common situation that occurs in
native extensions for CPython written in C++.

### Deadlocks

A deadlock can occur when more than one thread attempts to lock more than one
mutex, and two of the threads lock two of the mutexes in different orders. For
example, consider mutexes `mu1` and `mu2`, and threads T1 and T2, executing:

|    | T1                  | T2                 |
|--- | ------------------- | -------------------|
|1   | `mu1.lock()`{.good} | `mu2.lock()`{.good}|
|2   | `mu2.lock()`{.bad}  | `mu1.lock()`{.bad} |
|3   | `/* work */`        | `/* work */`       |
|4   | `mu2.unlock()`      | `mu1.unlock()`     |
|5   | `mu1.unlock()`      | `mu2.unlock()`     |

Now if T1 manages to lock `mu1` and T2 manages to lock `mu2` (as indicated in
green), then both threads will block while trying to lock the respective other
mutex (as indicated in red), but they are also unable to release the mutex that
they have locked (step 5).

**The problem** is that it is possible for one thread to attempt to lock `mu1`
and then `mu2`, and for another thread to attempt to lock `mu2` and then `mu1`.
Note that it does not matter if either mutex is unlocked at any intermediate
point; what matters is only the order of any attempt to *lock* the mutexes. For
example, the following, more complex series of operations is just as prone to
deadlock:

|    | T1                  | T2                 |
|--- | ------------------- | -------------------|
|1   | `mu1.lock()`{.good} | `mu1.lock()`{.good}|
|2   | waiting for T2      | `mu2.lock()`{.good}|
|3   | waiting for T2      | `/* work */`       |
|3   | waiting for T2      | `mu1.unlock()`     |
|3   | `mu2.lock()`{.bad}  | `/* work */`       |
|3   | `/* work */`        | `mu1.lock()`{.bad} |
|3   | `/* work */`        | `/* work */`       |
|4   | `mu2.unlock()`      | `mu1.unlock()`     |
|5   | `mu1.unlock()`      | `mu2.unlock()`     |

When the mutexes involved in a locking sequence are known at compile-time, then
avoiding deadlocks is &ldquo;merely&rdquo; a matter of arranging the lock
operations carefully so as to only occur in one single, fixed order. However, it
is also possible for mutexes to only be determined at runtime. A typical example
of this is a database where each row has its own mutex. An operation that
modifies two rows in a single transaction (e.g. &ldquo;transferring an amount
from one account to another&rdquo;) must lock two row mutexes, but the locking
order cannot be established at compile time. In this case, a dynamic
&ldquo;deadlock avoidance algorithm&rdquo; is needed. (In C++, `std::lock`
provides such an algorithm. An algorithm might use a non-blocking `try_lock`
operation on a mutex, which can either succeed or fail to lock the mutex, but
returns without blocking.)

Conceptually, one could also consider it a deadlock if _the same_ thread
attempts to lock a mutex that it has already locked (e.g. when some locked
operation accidentally recurses into itself): `mu.lock();`{.good}
`mu.lock();`{.bad} However, this is a slightly separate issue: Typical mutexes
are either of _recursive_ or _non-recursive_ kind. A recursive mutex allows
repeated locking and requires balanced unlocking. A non-recursive mutex can be
implemented more efficiently, and/but for efficiency reasons does not actually
guarantee a deadlock on second lock. Instead, the API simply forbids such use,
making it a precondition that the thread not already hold the mutex, with
undefined behaviour on violation.

### &ldquo;Once&rdquo; initialization

A common programming problem is to have an operation happen precisely once, even
if requested concurrently. While it is clear that we need to track in some
shared state somewhere whether the operation has already happened, it is worth
noting that this state only ever transitions, once, from `false` to `true`. This
is considerably simpler than a general shared state that can change values
arbitrarily. Next, we also need a mechanism for all but one thread to block
until the initialization has completed, which we can provide with a mutex. The
simplest solution just always locks the mutex:

```c++
// The "once" mechanism:
constinit absl::Mutex mu(absl::kConstInit);
constinit bool init_done = false;

// The operation of interest:
void f();

void InitOnceNaive() {
  absl::MutexLock lock(&mu);
  if (!init_done) {
    f();
    init_done = true;
  }
}
```

This works, but the efficiency-minded reader will observe that once the
operation has completed, all future lock contention on the mutex is
unnecessary. This leads to the (in)famous &ldquo;double-locking&rdquo;
algorithm, which was historically hard to write correctly. The idea is to check
the boolean *before* locking the mutex, and avoid locking if the operation has
already completed. However, accessing shared state concurrently when at least
one access is a write is prone to causing a data race and needs to be done
according to an appropriate concurrent programming model. In C++ we use atomic
variables:

```c++
// The "once" mechanism:
constinit absl::Mutex mu(absl::kConstInit);
constinit std::atomic<bool> init_done = false;

// The operation of interest:
void f();

void InitOnceWithFastPath() {
  if (!init_done.load(std::memory_order_acquire)) {
    absl::MutexLock lock(&mu);
    if (!init_done.load(std::memory_order_relaxed)) {
      f();
      init_done.store(true, std::memory_order_release);
    }
  }
}
```

Checking the flag now happens without holding the mutex lock, and if the
operation has already completed, we return immediately. After locking the mutex,
we need to check the flag again, since multiple threads can reach this point.

*Atomic details.* Since the atomic flag variable is accessed concurrently, we
have to think about the memory order of the accesses. There are two separate
cases: The first, outer check outside the mutex lock, and the second, inner
check under the lock. The outer check and the flag update form an
acquire/release pair: *if* the load sees the value `true` (which must have been
written by the store operation), then it also sees everything that happened
before the store, namely the operation `f()`. By contrast, the inner check can
use relaxed memory ordering, since in that case the mutex operations provide the
necessary ordering: if the inner load sees the value `true`, it happened after
the `lock()`, which happened after the `unlock()`, which happened after the
store.

The C++ standard library, and Abseil, provide a ready-made solution of this
algorithm called `std::call_once`/`absl::call_once`. (The interface is the same,
but the Abseil implementation is possibly better.)

```c++
// The "once" mechanism:
constinit absl::once_flag init_flag;

// The operation of interest:
void f();

void InitOnceWithCallOnce() {
  absl::call_once(once_flag, f);
}
```

Even though conceptually this is performing the same algorithm, this
implementation has some considerable advantages: The `once_flag` type is a small
and trivial, integer-like type and is trivially destructible. Not only does it
take up less space than a mutex, it also generates less code since it does not
have to run a destructor, which would need to be added to the program's global
destructor list.

The final clou comes with the C++ semantics of a `static` variable declared at
block scope: According to [[stmt.dcl]](https://eel.is/c++draft/stmt.dcl#3):

> Dynamic initialization of a block variable with static storage duration or
> thread storage duration is performed the first time control passes through its
> declaration; such a variable is considered initialized upon the completion of
> its initialization. [...] If control enters the declaration concurrently while
> the variable is being initialized, the concurrent execution shall wait for
> completion of the initialization.

This is saying that the initialization of a local, `static` variable precisely
has the &ldquo;once&rdquo; semantics that we have been discussing. We can
therefore write the above example as follows:

```c++
// The operation of interest:
void f();

void InitOnceWithStatic() {
  static int unused = (f(), 0);
}
```

This approach is by far the simplest and easiest, but the big difference is that
the mutex (or mutex-like object) in this implementation is no longer visible or
in the user&rsquo;s control. This is perfectly fine if the initializer is
simple, but if the initializer itself attempts to lock any other mutex
(including by initializing another static variable!), then we have no control
over the lock ordering!

Finally, you may have noticed the `constinit`s around the earlier code. Both
`constinit` and `constexpr` specifiers on a declaration mean that the variable
is *constant-initialized*, which means that no initialization is performed at
runtime (the initial value is already known at compile time). This in turn means
that a static variable guard mutex may not be needed, and static initialization
never blocks. The difference between the two is that a `constexpr`-specified
variable is also `const`, and a variable cannot be `constexpr` if it has a
non-trivial destructor. Such a destructor also means that the guard mutex is
needed after all, since the destructor must be registered to run at exit,
conditionally on initialization having happened.

## Python, CPython, GIL

With CPython, a Python program can call into native code. To this end, the
native code registers callback functions with the Python runtime via the CPython
API. In order to ensure that the internal state of the Python runtime remains
consistent, there is a single, shared mutex called the &ldquo;global interpreter
lock&rdquo;, or GIL for short. Upon entry of one of the user-provided callback
functions, the GIL is locked (or &ldquo;held&rdquo;), so that no other mutations
of the Python runtime state can occur until the native callback returns.

Many native extensions do not interact with the Python runtime for at least some
part of them, and so it is common for native extensions to _release_ the GIL, do
some work, and then reacquire the GIL before returning. Similarly, when code is
generally not holding the GIL but needs to interact with the runtime briefly, it
will first reacquire the GIL. The GIL is reentrant, and constructions to acquire
and subsequently release the GIL are common, and often don't worry about whether
the GIL is already held.

If the native code is written in C++ and contains local, `static` variables,
then we are now dealing with at least _two_ mutexes: the static variable guard
mutex, and the GIL from CPython.

A common problem in such code is an operation with &ldquo;only once&rdquo;
semantics that also ends up requiring the GIL to be held at some point. As per
the above description of &ldquo;once&rdquo;-style techniques, one might find a
static variable:

```c++
// CPython callback, assumes that the GIL is held on entry.
PyObject* InvokeWidget(PyObject* self) {
  static PyObject* impl = CreateWidget();
  return PyObject_CallOneArg(impl, self);
}
```

This seems reasonable, but bear in mind that there are two mutexes (the "guard
mutex" and "the GIL"), and we must think about the lock order. Otherwise, if the
callback is called from multiple threads, a deadlock may ensue.

Let us consider what we can see here: On entry, the GIL is already locked, and
we are locking the guard mutex. This is one lock order. Inside the initializer
`CreateWidget`, with both mutexes already locked, the function can freely access
the Python runtime.

However, it is entirely possible that `CreateWidget` will want to release the
GIL at one point and reacquire it later:

```c++
// Assumes that the GIL is held on entry.
// Ensures that the GIL is held on exit.
PyObject* CreateWidget() {
  // ...
  Py_BEGIN_ALLOW_THREADS  // releases GIL
  // expensive work, not accessing the Python runtime
  Py_END_ALLOW_THREADS    // acquires GIL, #!
  // ...
  return result;
}
```

Now we have a second lock order: the guard mutex is locked, and then the GIL is
locked (at `#!`). To see how this deadlocks, consider threads T1 and T2 both
having the runtime attempt to call `InvokeWidget`. T1 locks the GIL and
proceeds, locking the guard mutex and calling `CreateWidget`; T2 is blocked
waiting for the GIL. Then T1 releases the GIL to do &ldquo;expensive
work&rdquo;, and T2 awakes and locks the GIL. Now T2 is blocked trying to
acquire the guard mutex, but T1 is blocked reacquiring the GIL (at `#!`).

In other words: if we want to support &ldquo;once-called&rdquo; functions that
can arbitrarily release and reacquire the GIL, as is very common, then the only
lock order that we can ensure is: guard mutex first, GIL second.

To implement this, we must rewrite our code. Naively, we could always release
the GIL before a `static` variable with blocking initializer:

```c++
// CPython callback, assumes that the GIL is held on entry.
PyObject* InvokeWidget(PyObject* self) {
  Py_BEGIN_ALLOW_THREADS  // releases GIL
  static PyObject* impl = CreateWidget();
  Py_END_ALLOW_THREADS    // acquires GIL

  return PyObject_CallOneArg(impl, self);
}
```

But similar to the `InitOnceNaive` example above, this code cycles the GIL
(possibly descheduling the thread) even when the static variable has already
been initialized. If we want to avoid this, we need to abandon the use of a
static variable, since we do not control the guard mutex well enough. Instead,
we use an operation whose mutex locking is under our control, such as
`call_once`. For example:

```c++
// CPython callback, assumes that the GIL is held on entry.
PyObject* InvokeWidget(PyObject* self) {
  static constinit PyObject* impl = nullptr;
  static constinit std::atomic<bool> init_done = false;
  static constinit absl::once_flag init_flag;

  if (!init_done.load(std::memory_order_acquire)) {
    Py_BEGIN_ALLOW_THREADS                       // releases GIL
    absl::call_once(init_flag, [&]() {
      PyGILState_STATE s = PyGILState_Ensure();  // acquires GIL
      impl = CreateWidget();
      PyGILState_Release(s);                     // releases GIL
      init_done.store(true, std::memory_order_release);
    });
    Py_END_ALLOW_THREADS                         // acquires GIL
  }

  return PyObject_CallOneArg(impl, self);
}
```

The lock order is now always guard mutex first, GIL second. Unfortunately we
have to duplicate the &ldquo;double-checked done flag&rdquo;, effectively
leading to triple checking, because the flag state inside the `absl::once_flag`
is not accessible to the user. In other words, we cannot ask `init_flag` whether
it has been used yet.

However, we can perform one last, minor optimisation: since we assume that the
GIL is held on entry, and again when the initializing operation returns, the GIL
actually serializes access to our done flag variable, which therefore does not
need to be atomic. (The difference to the previous, atomic code may be small,
depending on the architecture. For example, on x86-64, acquire/release on a bool
is nearly free ([demo](https://godbolt.org/z/P9vYWf4fE)).)

```c++
// CPython callback, assumes that the GIL is held on entry, and indeed anywhere
// directly in this function (i.e. the GIL can be released inside CreateWidget,
// but must be reaqcuired when that call returns).
PyObject* InvokeWidget(PyObject* self) {
  static constinit PyObject* impl = nullptr;
  static constinit bool init_done = false;       // guarded by GIL
  static constinit absl::once_flag init_flag;

  if (!init_done) {
    Py_BEGIN_ALLOW_THREADS                       // releases GIL
                                                 // (multiple threads may enter here)
    absl::call_once(init_flag, [&]() {
                                                 // (only one thread enters here)
      PyGILState_STATE s = PyGILState_Ensure();  // acquires GIL
      impl = CreateWidget();
      init_done = true;                          // (GIL is held)
      PyGILState_Release(s);                     // releases GIL
    });

    Py_END_ALLOW_THREADS                         // acquires GIL
  }

  return PyObject_CallOneArg(impl, self);
}
```

## Debugging tips

*   Build with symbols.
*   <kbd>Ctrl</kbd>-<kbd>C</kbd> sends `SIGINT`, <kbd>Ctrl</kbd>-<kbd>\\</kbd>
    sends `SIGQUIT`. Both have their uses.
*   Useful `gdb` commands:
    *   `py-bt` prints a Python backtrace if you are in a Python frame.
    *   `thread apply all bt 10` prints the top-10 frames for each thread. A
        full backtrace can be prohibitively expensive, and the top few frames
        are often good enough.
    *   `p PyGILState_Check()` shows whether a thread is holding the GIL. For
        all threads, run `thread apply all p PyGILState_Check()` to find out
        which thread is holding the GIL.
    *   The `static` variable guard mutex is accessed with functions like
        `cxa_guard_acquire` (though this depends on ABI details and can vary).
        The guard mutex itself contains information about which thread is
        currently holding it.

## Links

*   Article on
    [double-checked locking](https://preshing.com/20130930/double-checked-locking-is-fixed-in-cpp11/)
*   [The Deadlock Empire](https://deadlockempire.github.io/), hands-on exercises
    to construct deadlocks
