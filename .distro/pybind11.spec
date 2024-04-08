Name:           pybind11
Version:        0.0.0
Release:        %autorelease
Summary:        Seamless operability between C++11 and Python

# TODO: https://github.com/pybind/pybind11/issues/5093
License:        BSD
URL:            https://github.com/pybind/pybind11
BuildArch:      noarch
Source0:        https://github.com/pybind/pybind11/archive/v%{version}/%{name}-%{version}.tar.gz

BuildRequires:  python3-devel
BuildRequires:  cmake
BuildRequires:  ninja-build
BuildRequires:  gcc
BuildRequires:  gcc-c++
# Test dependencies from tests/requirements.txt
# Cannot include the file because requirements are too constrained
BuildRequires:  python3dist(build)
BuildRequires:  python3dist(numpy)
BuildRequires:  python3dist(pytest)
BuildRequires:  python3dist(pytest-timeout)
BuildRequires:  python3dist(scipy)

%global _description %{expand:
pybind11 is a lightweight header-only library that exposes C++ types
in Python and vice versa, mainly to create Python bindings of existing
C++ code.
}

%description %_description

%package devel
Summary:        Development files for pybind11
BuildRequires:  pybind11-devel
Provides:       pybind11-static = %{version}-%{release}
%description devel %_description

Development files.

%package -n python3-pybind11
Summary:        %{summary}
BuildRequires:  pybind11-devel
%description -n python3-pybind11 %_description


%prep
%autosetup -n pybind11-%{version}
# Remove cmake and ninja from buildrequires
sed -i -E 's/,?\s*"cmake[^"]*"//' pyproject.toml
sed -i -E 's/,?\s*"ninja[^"]*"//' pyproject.toml


%generate_buildrequires
%pyproject_buildrequires


%build
%pyproject_wheel


%install
%pyproject_install
%pyproject_save_files pybind11


%check
%pytest


%files devel
%license LICENSE
%doc README.rst
%{_includedir}/pybind11/
%{_datadir}/cmake/pybind11/
%{_bindir}/pybind11-config
%{_datadir}/pkgconfig/pybind11.pc

%files -n python3-pybind11 -f %{pyproject_files}


%changelog
%autochangelog
