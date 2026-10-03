#include <string>

struct Pet {
    explicit Pet(const std::string &name) : name(name) {}
    std::string name;
};
