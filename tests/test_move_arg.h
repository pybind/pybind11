#pragma once
#include <memory>
#include <vector>
#include <iostream>

class Item;
std::ostream& operator<<(std::ostream& os, const Item& item);

/** Item class requires unique instances, i.e. only supports move construction */
class Item
{
public:
  Item(int value) : value_(std::make_unique<int>(value)) {
    std::cout<< "new " << *this << "\n";
  }
  ~Item() {
    std::cout << "destroy " << *this << "\n";
  }
  Item(const Item&) = delete;
  Item(Item&& other) {
    std::cout << "move " << other << " -> ";
    value_ = std::move(other.value_);
    std::cout << *this << "\n";
  }

  std::unique_ptr<int> value_;
};

std::ostream& operator<<(std::ostream& os, const Item& item) {
  os << "item " << &item << "(";
  if (item.value_) os << *item.value_;
  os << ")";
  return os;
}
