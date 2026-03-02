#pragma once

#include <string>

#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/map.h"

namespace docc {
namespace qant {

/**
 * @brief Qant implementation with automatic memory transfers
 * Uses QANT native computing toolkit with automatic host-device data transfers
 */
inline sdfg::data_flow::ImplementationType ImplementationType_QANT{"qant"};

} // namespace qant
} // namespace docc
