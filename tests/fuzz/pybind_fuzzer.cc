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

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#ifndef PYBIND11_FUZZER_STANDALONE
#    include <fuzzer/FuzzedDataProvider.h>
#endif

#include <cstdint>
#include <string>

namespace py = pybind11;

namespace {

// Same length split as FuzzedDataProvider::ConsumeRandomLengthString. The
// checked-in corpus is framed for this split. OSS-Fuzz still consumes with
// FuzzedDataProvider below; replay an OSS-Fuzz artifact with libFuzzer.
std::string
consume_string(const uint8_t *&data, size_t &size) {
    if (size == 0) {
        return {};
    }
    const auto selector = static_cast<size_t>(*data);
    ++data;
    --size;
    const size_t length = (selector * (size + 1)) / 255;
    const size_t n = length < size ? length : size;
    std::string out(reinterpret_cast<const char *>(data), n);
    data += n;
    size -= n;
    return out;
}

int
exercise(const std::string &code, const std::string &attr_name, const std::string &decimal_text,
         const std::string &decimal_attr, const std::string &text) {
    static py::scoped_interpreter guard{};
    try {
        auto locals = py::dict();
        py::exec(code, py::globals(), locals);
    } catch (const py::error_already_set &) {
    }

    try {
        py::object os = py::module_::import("os");
        py::object looked_up = os.attr(attr_name.c_str());
        (void) looked_up;
    } catch (const py::error_already_set &) {
    }

    try {
        py::tuple args = py::make_tuple(decimal_text, py::none());
        (void) args;
        py::object Decimal = py::module_::import("decimal").attr("Decimal");
        py::object pi = Decimal(decimal_text);
        py::object called = pi.attr(decimal_attr.c_str())();
        (void) called;
    } catch (const py::error_already_set &) {
    }

    try {
        py::object obj = py::str(text);
        (void) obj;
    } catch (const py::error_already_set &) {
    }
    return 0;
}

} // namespace

extern "C" int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
#ifdef PYBIND11_FUZZER_STANDALONE
    return exercise(consume_string(data, size), consume_string(data, size), consume_string(data, size),
                    consume_string(data, size), consume_string(data, size));
#else
    FuzzedDataProvider fdp(data, size);
    return exercise(fdp.ConsumeRandomLengthString(), fdp.ConsumeRandomLengthString(),
                    fdp.ConsumeRandomLengthString(), fdp.ConsumeRandomLengthString(),
                    fdp.ConsumeRandomLengthString());
#endif
}
