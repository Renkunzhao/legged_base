#pragma once
// yaml_utils.hpp
// Minimal YAML read helpers with "missing key -> throw" diagnostics.

#include <yaml-cpp/yaml.h>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace legged_base::yaml {

namespace detail {

inline std::string toString(std::string_view sv) { return std::string(sv); }

inline std::runtime_error missingKey(std::string_view ctx, std::string_view key) {
  std::ostringstream oss;
  oss << "[YAML][" << ctx << "] missing required key: `" << key << "`";
  return std::runtime_error(oss.str());
}

inline std::runtime_error badType(std::string_view ctx,
                                 std::string_view key,
                                 std::string_view expected,
                                 const YAML::Exception& e) {
  std::ostringstream oss;
  oss << "[YAML][" << ctx << "] bad type for key: `" << key << "`, expected "
      << expected << ". (" << e.what() << ")";
  return std::runtime_error(oss.str());
}

}  // namespace detail

// ---------- Existence helpers ----------

inline bool hasKey(const YAML::Node& node, std::string_view key) {
  return node && node[detail::toString(key)];
}

inline YAML::Node requireNode(const YAML::Node& node,
                              std::string_view key,
                              std::string_view ctx) {
  if (!node || !node[detail::toString(key)]) {
    throw detail::missingKey(ctx, key);
  }
  return node[detail::toString(key)];
}

// ---------- Typed reads (required) ----------

template <typename T>
inline T require(const YAML::Node& node,
                 std::string_view key,
                 std::string_view ctx) {
  static_assert(!std::is_reference_v<T>, "T must be a value type");
  YAML::Node n = requireNode(node, key, ctx);
  try {
    return n.as<T>();
  } catch (const YAML::Exception& e) {
    // Note: type name is not easily obtainable portably; keep it minimal.
    throw detail::badType(ctx, key, "requested type", e);
  }
}

// Small convenience wrappers (optional)
inline double requireDouble(const YAML::Node& node, std::string_view key, std::string_view ctx) {
  try {
    return require<double>(node, key, ctx);
  } catch (const std::runtime_error& e) {
    throw;  // keep stack/message
  }
}

inline int requireInt(const YAML::Node& node, std::string_view key, std::string_view ctx) {
  return require<int>(node, key, ctx);
}

inline bool requireBool(const YAML::Node& node, std::string_view key, std::string_view ctx) {
  return require<bool>(node, key, ctx);
}

inline std::string requireString(const YAML::Node& node, std::string_view key, std::string_view ctx) {
  return require<std::string>(node, key, ctx);
}

// ---------- Typed reads (optional) ----------

template <typename T>
inline std::optional<T> optional(const YAML::Node& node, std::string_view key) {
  if (!node) return std::nullopt;
  YAML::Node n = node[detail::toString(key)];
  if (!n) return std::nullopt;
  try {
    return n.as<T>();
  } catch (...) {
    return std::nullopt;  // minimal behavior; use require<T> if you want error
  }
}

template <typename T>
inline T readOr(const YAML::Node& node,
                std::string_view key,
                const T& default_value,
                std::string_view ctx) {
  if (!hasKey(node, key)) return default_value;
  // If present but wrong type, throw with a useful message:
  return require<T>(node, key, ctx);
}

// ---------- Minimal value checks (optional, but handy) ----------

inline void requirePositive(double v, std::string_view name, std::string_view ctx) {
  if (!(v > 0.0)) {
    std::ostringstream oss;
    oss << "[YAML][" << ctx << "] `" << name << "` must be > 0, got " << v;
    throw std::runtime_error(oss.str());
  }
}

inline void requireNonNegative(double v, std::string_view name, std::string_view ctx) {
  if (v < 0.0) {
    std::ostringstream oss;
    oss << "[YAML][" << ctx << "] `" << name << "` must be >= 0, got " << v;
    throw std::runtime_error(oss.str());
  }
}

}  // namespace legged_base::yaml
