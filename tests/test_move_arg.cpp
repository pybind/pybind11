#include "test_move_arg.h"
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <sstream>

namespace py = pybind11;

#if 1
template <typename T>
class my_ptr {
public:
	my_ptr(T* p = nullptr) : ptr_(p) {}
	my_ptr(my_ptr<T>&& other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }
	~my_ptr() { delete ptr_; }
	my_ptr<T>& operator=(my_ptr<T>&& other) { ptr_ = other.ptr_; other.ptr_ = nullptr; return *this; }
	const T* get() const { return ptr_; }
	const T* verbose_get() const {
		std::cout << " [" << ptr_ << "] "; return ptr_;
	}
private:
	T* ptr_;
};
PYBIND11_DECLARE_HOLDER_TYPE(T, my_ptr<T>)
namespace pybind11 { namespace detail {
    template <typename T>
    struct holder_helper<my_ptr<T>> { // <-- specialization
        static const T *get(const my_ptr<T> &p) { return p.verbose_get(); }
    };
}}
#else
template <typename T>
using my_ptr = std::unique_ptr<T>;
#endif

PYBIND11_MODULE(test_move_arg, m) {
	py::class_<Item, my_ptr<Item>>(m, "Item")
		.def(py::init<int>(), py::call_guard<py::scoped_ostream_redirect>())
		.def("__repr__", [](const Item& item) {
			std::stringstream ss;
			ss << "py " << item;
			return ss.str();
		}, py::call_guard<py::scoped_ostream_redirect>());

	m.def("access", [](const Item& item) {
		std::cout << "access " << item << "\n";
	}, py::call_guard<py::scoped_ostream_redirect>());

#if 0 // rvalue arguments fail during compilation
	m.def("consume", [](Item&& item) {
		std::cout << "consume " << item << "\n  ";
		Item sink(std::move(item));
		std::cout << "  old: " << item << "\n  new: " << sink << "\n";
	}, py::call_guard<py::scoped_ostream_redirect>());
#endif

	m.def("consume", [](my_ptr<Item>&& item) {
		std::cout << "consume " << *item.get() << "\n";
		my_ptr<Item> sink(std::move(item));
		std::cout << "  old: " << item.get() << "\n  new: " << *sink.get() << "\n";
	}, py::call_guard<py::scoped_ostream_redirect>());

	m.def("consume_str", [](std::string&& s) { std::string o(std::move(s)); });
}
