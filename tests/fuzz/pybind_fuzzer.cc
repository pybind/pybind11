/* Copyright 2023 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <fuzzer/FuzzedDataProvider.h>
#include <iostream>
#include <string>
#include "pybind11/embed.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  FuzzedDataProvider fdp(data, size);

  static py::scoped_interpreter guard{};
  try {
    auto locals = py::dict();
    py::exec(fdp.ConsumeRandomLengthString().c_str(), py::globals(), locals);
  } catch (pybind11::error_already_set &e) {
  }

  try {
    py::object os = py::module_::import("os");
    py::object makedirs = os.attr(fdp.ConsumeRandomLengthString().c_str());
  } catch (py::error_already_set &e) {
  }

  try {
    py::tuple args =
        py::make_tuple(fdp.ConsumeRandomLengthString().c_str(), py::none());
    py::object Decimal = py::module_::import("decimal").attr("Decimal");
    py::object pi = Decimal(fdp.ConsumeRandomLengthString().c_str());
    py::object exp_pi = pi.attr(fdp.ConsumeRandomLengthString().c_str())();
  } catch (py::error_already_set &e) {
  }

  try {
    py::object obj = py::str(fdp.ConsumeRandomLengthString().c_str());
  } catch (py::error_already_set &e) {
  }
  return 0;
}
