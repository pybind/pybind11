#include "pybind11_tests.h"

namespace pybind11_tests {
namespace class_custom_type_caster_mix {

class NumberStore {
private:
    int stored;

public:
    explicit NumberStore(int num) : stored{num} {}
    int Get() const { return stored; }
};

} // namespace class_custom_type_caster_mix
} // namespace pybind11_tests

namespace pybind11 {
namespace detail {

using namespace pybind11_tests::class_custom_type_caster_mix;

template <>
struct type_caster<NumberStore> {
public:
    PYBIND11_TYPE_CASTER(NumberStore,
                         _("pybind11_tests::class_custom_type_caster_mix::NumberStore"));

    bool load(handle /*handle*/, bool /*convert*/) {
        value = NumberStore(5);
        return true;
    }
};

} // namespace detail
} // namespace pybind11

TEST_SUBMODULE(class_custom_type_caster_mix, m) {
    using namespace pybind11_tests::class_custom_type_caster_mix;

    py::class_<NumberStore>(m, "NumberStore").def(py::init<int>()).def("Get", &NumberStore::Get);
}
