#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#ifdef TENSORIUM_USE_HDF5
#include <hdf5.h>
#endif

namespace tensorium {
namespace io {

#ifdef TENSORIUM_USE_HDF5
using hdf5_handle_t = hid_t;
#else
using hdf5_handle_t = int;
#endif

inline constexpr bool hdf5_available() noexcept {
#ifdef TENSORIUM_USE_HDF5
    return true;
#else
    return false;
#endif
}

#ifdef TENSORIUM_USE_HDF5
namespace detail {

inline void close_if_valid(hid_t &handle, herr_t (*closer)(hid_t)) {
    if (handle >= 0) {
        closer(handle);
        handle = -1;
    }
}

inline hid_t make_utf8_string_type(std::size_t size) {
    hid_t type = H5Tcopy(H5T_C_S1);
    if (type < 0)
        return -1;
    const auto width = static_cast<size_t>(std::max<std::size_t>(1, size));
    if (H5Tset_size(type, width) < 0 || H5Tset_cset(type, H5T_CSET_UTF8) < 0 ||
        H5Tset_strpad(type, H5T_STR_NULLTERM) < 0) {
        H5Tclose(type);
        return -1;
    }
    return type;
}

inline hid_t create_scalar_space() { return H5Screate(H5S_SCALAR); }

} // namespace detail
#endif

class HDF5File {
  public:
    HDF5File() = default;
    explicit HDF5File(const std::string &path) { (void)open(path); }

    HDF5File(const HDF5File &) = delete;
    HDF5File &operator=(const HDF5File &) = delete;

    HDF5File(HDF5File &&other) noexcept { *this = std::move(other); }
    HDF5File &operator=(HDF5File &&other) noexcept {
        if (this == &other)
            return *this;
        close();
        path_ = std::move(other.path_);
        handle_ = other.handle_;
        other.handle_ = -1;
        return *this;
    }

    ~HDF5File() { close(); }

    bool open(const std::string &path) {
        close();
        path_ = path;
#ifdef TENSORIUM_USE_HDF5
        handle_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        return handle_ >= 0;
#else
        (void)path;
        return false;
#endif
    }

    void close() {
#ifdef TENSORIUM_USE_HDF5
        detail::close_if_valid(handle_, H5Fclose);
#endif
        handle_ = -1;
    }

    [[nodiscard]] bool is_open() const noexcept {
#ifdef TENSORIUM_USE_HDF5
        return handle_ >= 0;
#else
        return false;
#endif
    }

    [[nodiscard]] hdf5_handle_t id() const noexcept { return handle_; }
    [[nodiscard]] const std::string &path() const noexcept { return path_; }

  private:
    std::string   path_;
    hdf5_handle_t handle_ = -1;
};

inline bool write_string_attribute(hdf5_handle_t object, const std::string &name,
                                   const std::string &value) {
#ifdef TENSORIUM_USE_HDF5
    if (object < 0)
        return false;
    hid_t type = detail::make_utf8_string_type(value.size() + 1);
    if (type < 0)
        return false;
    hid_t space = detail::create_scalar_space();
    if (space < 0) {
        H5Tclose(type);
        return false;
    }
    hid_t attr = H5Acreate2(object, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(space);
        H5Tclose(type);
        return false;
    }
    const bool ok = H5Awrite(attr, type, value.c_str()) >= 0;
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
    return ok;
#else
    (void)object;
    (void)name;
    (void)value;
    return false;
#endif
}

inline bool write_scalar_attribute(hdf5_handle_t object, const std::string &name, double value) {
#ifdef TENSORIUM_USE_HDF5
    if (object < 0)
        return false;
    hid_t space = detail::create_scalar_space();
    if (space < 0)
        return false;
    hid_t attr =
        H5Acreate2(object, name.c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(space);
        return false;
    }
    const bool ok = H5Awrite(attr, H5T_NATIVE_DOUBLE, &value) >= 0;
    H5Aclose(attr);
    H5Sclose(space);
    return ok;
#else
    (void)object;
    (void)name;
    (void)value;
    return false;
#endif
}

inline bool write_scalar_attribute(hdf5_handle_t object, const std::string &name,
                                   std::uint64_t value) {
#ifdef TENSORIUM_USE_HDF5
    if (object < 0)
        return false;
    hid_t space = detail::create_scalar_space();
    if (space < 0)
        return false;
    hid_t attr =
        H5Acreate2(object, name.c_str(), H5T_NATIVE_UINT64, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(space);
        return false;
    }
    const bool ok = H5Awrite(attr, H5T_NATIVE_UINT64, &value) >= 0;
    H5Aclose(attr);
    H5Sclose(space);
    return ok;
#else
    (void)object;
    (void)name;
    (void)value;
    return false;
#endif
}

inline bool write_vector_dataset(hdf5_handle_t file, const std::string &name,
                                 const std::vector<double> &values) {
#ifdef TENSORIUM_USE_HDF5
    if (file < 0)
        return false;
    const hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    if (space < 0)
        return false;
    hid_t dataset =
        H5Dcreate2(file, name.c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT,
                   H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    const bool ok =
        values.empty() || H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                   values.data()) >= 0;
    H5Dclose(dataset);
    H5Sclose(space);
    return ok;
#else
    (void)file;
    (void)name;
    (void)values;
    return false;
#endif
}

inline bool write_matrix_dataset(hdf5_handle_t file, const std::string &name, std::size_t rows,
                                 std::size_t cols, const std::vector<double> &values) {
#ifdef TENSORIUM_USE_HDF5
    if (file < 0 || values.size() != rows * cols)
        return false;
    const hsize_t dims[2] = {static_cast<hsize_t>(rows), static_cast<hsize_t>(cols)};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    if (space < 0)
        return false;
    hid_t dataset =
        H5Dcreate2(file, name.c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT,
                   H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    const bool ok =
        values.empty() || H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                   values.data()) >= 0;
    H5Dclose(dataset);
    H5Sclose(space);
    return ok;
#else
    (void)file;
    (void)name;
    (void)rows;
    (void)cols;
    (void)values;
    return false;
#endif
}

class HDF5AppendTable {
  public:
    HDF5AppendTable() = default;
    HDF5AppendTable(const HDF5AppendTable &) = delete;
    HDF5AppendTable &operator=(const HDF5AppendTable &) = delete;

    HDF5AppendTable(HDF5AppendTable &&other) noexcept { *this = std::move(other); }
    HDF5AppendTable &operator=(HDF5AppendTable &&other) noexcept {
        if (this == &other)
            return *this;
        close();
        path_ = std::move(other.path_);
        columns_ = std::move(other.columns_);
        datasets_ = std::move(other.datasets_);
        handle_ = other.handle_;
        other.handle_ = -1;
        other.datasets_.clear();
        other.columns_.clear();
        return *this;
    }

    ~HDF5AppendTable() { close(); }

    bool open(const std::string &path, std::vector<std::string> columns,
              std::size_t chunk_size = 256) {
        close();
        path_ = path;
        columns_ = std::move(columns);
        if (columns_.empty())
            return false;
#ifdef TENSORIUM_USE_HDF5
        handle_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (handle_ < 0)
            return false;
        datasets_.reserve(columns_.size());
        for (const auto &column : columns_) {
            const hsize_t dims[1] = {0};
            const hsize_t maxdims[1] = {H5S_UNLIMITED};
            const hsize_t chunk_dims[1] = {static_cast<hsize_t>(std::max<std::size_t>(1, chunk_size))};
            hid_t space = H5Screate_simple(1, dims, maxdims);
            if (space < 0) {
                close();
                return false;
            }
            hid_t props = H5Pcreate(H5P_DATASET_CREATE);
            if (props < 0) {
                H5Sclose(space);
                close();
                return false;
            }
            if (H5Pset_chunk(props, 1, chunk_dims) < 0) {
                H5Pclose(props);
                H5Sclose(space);
                close();
                return false;
            }
            hid_t dataset = H5Dcreate2(handle_, column.c_str(), H5T_NATIVE_DOUBLE, space,
                                       H5P_DEFAULT, props, H5P_DEFAULT);
            H5Pclose(props);
            H5Sclose(space);
            if (dataset < 0) {
                close();
                return false;
            }
            datasets_.push_back(dataset);
        }
        (void)write_string_attribute(handle_, "tensorium_format", "columnar_append_v1");
        return true;
#else
        (void)chunk_size;
        return false;
#endif
    }

    void close() {
#ifdef TENSORIUM_USE_HDF5
        for (auto &dataset : datasets_)
            detail::close_if_valid(dataset, H5Dclose);
        detail::close_if_valid(handle_, H5Fclose);
#endif
        datasets_.clear();
        columns_.clear();
        handle_ = -1;
    }

    [[nodiscard]] bool is_open() const noexcept {
#ifdef TENSORIUM_USE_HDF5
        return handle_ >= 0 && !datasets_.empty();
#else
        return false;
#endif
    }

    bool append_row(const std::vector<double> &values) {
#ifdef TENSORIUM_USE_HDF5
        if (!is_open() || values.size() != datasets_.size())
            return false;
        for (std::size_t idx = 0; idx < datasets_.size(); ++idx) {
            hid_t dataset = datasets_[idx];
            hid_t file_space = H5Dget_space(dataset);
            if (file_space < 0)
                return false;
            hsize_t dims[1] = {0};
            H5Sget_simple_extent_dims(file_space, dims, nullptr);
            const hsize_t next_dims[1] = {dims[0] + 1};
            if (H5Dset_extent(dataset, next_dims) < 0) {
                H5Sclose(file_space);
                return false;
            }
            H5Sclose(file_space);
            file_space = H5Dget_space(dataset);
            if (file_space < 0)
                return false;
            const hsize_t start[1] = {dims[0]};
            const hsize_t count[1] = {1};
            if (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr) <
                0) {
                H5Sclose(file_space);
                return false;
            }
            hid_t mem_space = H5Screate_simple(1, count, nullptr);
            if (mem_space < 0) {
                H5Sclose(file_space);
                return false;
            }
            const double value = values[idx];
            const bool ok =
                H5Dwrite(dataset, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, &value) >=
                0;
            H5Sclose(mem_space);
            H5Sclose(file_space);
            if (!ok)
                return false;
        }
        H5Fflush(handle_, H5F_SCOPE_LOCAL);
        return true;
#else
        (void)values;
        return false;
#endif
    }

    [[nodiscard]] const std::string &path() const noexcept { return path_; }

  private:
    std::string                path_;
    std::vector<std::string>   columns_;
    std::vector<hdf5_handle_t> datasets_;
    hdf5_handle_t              handle_ = -1;
};

} // namespace io
} // namespace tensorium
