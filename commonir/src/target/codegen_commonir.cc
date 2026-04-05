/*!
 * \file target/codegen.cc
 */

#include "codegen_commonir.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/fill.h"
#include "../op/gemm.h"
#include "../op/reduce.h"
#include "../op/region.h"
#include "arith/pattern_match.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <cassert>
#include <cmath>
#include <elf.h>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

using ffi::Array;
using ffi::String;

template <typename T>
inline void CodeGenTileLangCOMMONIR::PrintBinary(const T *op, const char *opstr,
                                                 std::ostream &os) {
  auto PrintOp = [op, &os, this](auto Operand) {
    std::ostringstream tmpos;
    if (Operand.template as<tvm::tir::IntImmNode>() ||
        Operand.template as<tvm::tir::FloatImmNode>() ||
        Operand.template as<tvm::tir::VarNode>()) {
      PrintExpr(Operand, tmpos << "%");
    } else {
      std::string op_name = SSAGetID(PrintExpr(Operand), Operand->dtype);
      tmpos << "%" << op_name;
    }
    return tmpos.str();
  };
  if (op->dtype.lanes() == 1) {
    // left op
    os << "arith." << opstr << " ";
    os << PrintOp(op->a);
    os << ", ";
    // right op
    os << PrintOp(op->b);
    os << " : ";
    PrintType(op->a->dtype, os);
  } else {
    os << "<<<invalid-op-dtype-lanes-not-one: %" << opstr << ">>>\n";
  }
}

// for future use
String GetAddressSpace(String address_space) {
  if (address_space == "global")
    return "global";
  else if (address_space == "shared")
    return "shared";
  else if (address_space == "shared.dyn")
    return "shared";
  else if (address_space == "local.fragment")
    return "local";
  return "unknown";
}

bool IsEqual(Array<PrimExpr> a, Array<PrimExpr> b) {
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (!(a[i].same_as(b[i])))
      return false;
  }
  return true;
}

bool AllZero(Array<PrimExpr> a) {
  for (PrimExpr pe : a) {
    if (!is_zero(pe))
      return false;
  }
  return true;
}

bool IsStaticUnitExtent(const PrimExpr &expr) {
  if (const int64_t *expr_int = as_const_int(expr)) {
    return *expr_int == 1;
  }
  return false;
}

Array<PrimExpr> GetSubviewResultShape(Array<PrimExpr> shape) {
  Array<PrimExpr> result_shape;
  for (const PrimExpr &extent : shape) {
    if (!IsStaticUnitExtent(extent)) {
      result_shape.push_back(extent);
    }
  }
  return result_shape;
}

Array<PrimExpr> GetSubviewResultStride(Array<PrimExpr> shape,
                                       Array<PrimExpr> stride) {
  ICHECK(stride.empty() || stride.size() == shape.size())
      << "Subview shape and stride rank mismatch";
  Array<PrimExpr> result_stride;
  for (int i = 0; i < shape.size(); ++i) {
    if (!IsStaticUnitExtent(shape[i]) && !stride.empty()) {
      result_stride.push_back(stride[i]);
    }
  }
  return result_stride;
}

std::vector<unsigned long> GetStrideFromShape(Array<tvm::PrimExpr> shape) {
  std::vector<unsigned long> strides;
  unsigned long total_size = 1;
  std::vector<int> shape_int;
  for (PrimExpr s : shape) {
    if (auto s_int = as_const_int(s)) {
      total_size *= *s_int;
      shape_int.push_back(*s_int);
    }
  }
  for (int i = 0; i < shape.size(); i++) {
    total_size /= shape_int[i];
    strides.push_back(total_size);
  }
  return strides;
}

String GetBufferStrides(Buffer buffer) {
  Array<PrimExpr> shape = buffer->shape;
  std::vector<unsigned long> strides;
  int dim = buffer->shape.size();
  if (buffer->strides.empty()) {
    strides = GetStrideFromShape(shape);
  } else {
    for (PrimExpr stride : buffer->strides) {
      if (auto stride_int = as_const_int(stride)) {
        strides.push_back(*stride_int);
      }
    }
  }
  String res = "[";
  for (int i = 0; i < dim; i++) {
    if (i > 0)
      res = res + ", ";
    res = res + std::to_string(strides[i]);
  }
  res = res + "]";
  return res;
}

static std::vector<int> getBroadcastDim(Array<PrimExpr> &buffer_shape0,
                                        Array<PrimExpr> &buffer_shape1) {
  assert(buffer_shape0.size() == buffer_shape1.size());
  std::vector<int> dims;
  for (int i = 0; i < buffer_shape0.size(); i++) {
    if (*as_const_int(buffer_shape0[i]) == 1 &&
        *as_const_int(buffer_shape1[i]) != 1) {
      dims.emplace_back(i);
    }
    if (*as_const_int(buffer_shape0[i]) != 1 &&
        *as_const_int(buffer_shape1[i]) == 1) {
      dims.emplace_back(i);
    }
    assert(*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i]));
  }
  return dims;
}

static std::string broadcastAttrCodegen(Array<PrimExpr> &buffer_shape0,
                                        Array<PrimExpr> &buffer_shape1) {
  if (buffer_shape0.empty() || buffer_shape1.empty()) {
    return "";
  }
  std::vector<int> broadcastDims;
  if (buffer_shape0.size() && buffer_shape1.size()) {
    broadcastDims = getBroadcastDim(buffer_shape0, buffer_shape1);
  }
  std::ostringstream temp;
  if (broadcastDims.size()) {
    temp << " = [";
    for (auto dim : broadcastDims) {
      temp << dim;
      if (dim != broadcastDims.back()) {
        temp << ", ";
      }
    }
    temp << "]";
  }
  return temp.str();
}

void PrintBufferMap(const Map<Var, Buffer> &buffer_map) {
  for (const auto &kv : buffer_map) {
    const Var &var = kv.first;
    const Buffer &buffer = kv.second;

    LOG(INFO) << "Var: " << var->name_hint << ", Buffer Name: " << buffer->name
              << ", Buffer Shape: " << buffer->shape
              << ", Buffer Dtype: " << buffer->dtype;
  }
}

std::string GetCastOp(DataType src_type, DataType dst_type) {
  bool srcIsFloat = src_type.is_float() || src_type.is_bfloat16();
  bool srcIsInt = src_type.is_int();
  bool srcIsUInt = src_type.is_uint();
  bool targetIsFloat = dst_type.is_float() || dst_type.is_bfloat16();
  bool targetIsInt = dst_type.is_int();
  bool targetIsUInt = dst_type.is_uint();
  if (srcIsFloat && targetIsInt) {
    return "arith.fptosi";
  } else if (srcIsFloat && targetIsUInt) {
    return "arith.fptoui";
  } else if (srcIsInt && targetIsFloat) {
    return "arith.sitofp";
  } else if (srcIsUInt && targetIsFloat) {
    return "arith.uitofp";
  } else if (targetIsInt) {
    if (dst_type.bits() > src_type.bits()) {
      return "arith.extsi";
    } else {
      return "arith.trunci";
    }
  } else if (targetIsUInt) {
    if (dst_type.bits() > src_type.bits()) {
      return "arith.extui";
    } else {
      return "arith.trunci";
    }
  } else if (targetIsFloat) {
    if (dst_type.bits() > src_type.bits()) {
      return "arith.extf";
    } else {
      return "arith.truncf";
    }
  }
}

CodeGenTileLangCOMMONIR::CodeGenTileLangCOMMONIR() {}

void CodeGenTileLangCOMMONIR::PrintFuncPrefix(std::ostream &os) {}

std::string CodeGenTileLangCOMMONIR::Finish() {
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const tir::ForNode *op) {
  if (op->kind == tir::ForKind::kParallel) {
    assert(op->extent.dtype().is_int() || op->extent.dtype().is_uint());
    assert(op->min.dtype() == op->extent.dtype());
    std::string upperBoundId =
        SSAGetID(PrintExpr(arith::Analyzer().Simplify(op->extent + op->min)),
                 op->extent->dtype);

    std::ostringstream temp;
    temp << "arith.index_cast %" << upperBoundId << ": ";
    PrintType(op->extent.dtype(), temp);
    temp << " to index";
    std::string upperBoundId_index = SSAGetID(temp.str(), op->extent->dtype);

    std::string lowerBoundId = SSAGetID(PrintExpr(op->min), op->min->dtype);
    temp.str("");
    temp.clear();
    temp << "arith.index_cast %" << lowerBoundId << ": ";
    PrintType(op->min.dtype(), temp);
    temp << " to index";
    std::string lowerBoundId_index = SSAGetID(temp.str(), op->min->dtype);

    auto stepNode = std::make_unique<IntImm>(op->min.dtype(), 1);
    auto stepId = SSAGetID(PrintExpr(*stepNode), stepNode->dtype());
    temp.str("");
    temp.clear();
    temp << "arith.index_cast %" << stepId << ": ";
    PrintType(op->min.dtype(), temp);
    temp << " to index";
    std::string stepId_index = SSAGetID(temp.str(), stepNode->dtype());

    PrintIndent();

    std::string vid =
        SSAGetID(AllocVarID(op->loop_var.get()), op->loop_var->dtype);
    stream << "scf.parallel"
           << " (%" << vid << "_index) = (%" << lowerBoundId_index << ") to (%"
           << upperBoundId_index << ") step (%" << stepId_index << ") ";
    stream << " {\n";

    int for_scope = BeginScope();
    PrintIndent();
    stream << "%" << vid << "= arith.index_cast %" << vid
           << "_index: index to ";
    PrintType(op->loop_var->dtype, stream);
    stream << "\n";
    PrintStmt(op->body);
    this->EndScope(for_scope);
    PrintIndent();
    stream << "}\n";
  } else if (op->kind == tir::ForKind::kSerial) {
    std::string upperBoundId =
        SSAGetID(PrintExpr(arith::Analyzer().Simplify(op->extent + op->min)),
                 op->extent->dtype);
    assert(op->extent.dtype().is_int() || op->extent.dtype().is_uint());
    assert(op->min.dtype() == op->extent.dtype());
    std::string vid =
        SSAGetID(AllocVarID(op->loop_var.get()), op->loop_var->dtype);
    std::string lowerBoundId = SSAGetID(PrintExpr(op->min), op->min->dtype);
    std::string extentId = SSAGetID(PrintExpr(op->extent), op->extent->dtype);
    auto stepNode = std::make_unique<IntImm>(op->min.dtype(), 1);
    auto stepId = SSAGetID(PrintExpr(*stepNode), stepNode->dtype());
    PrintIndent();
    stream << "scf.for"
           << " %" << vid << " = %" << lowerBoundId << " to %" << upperBoundId
           << " step %" << stepId << " : ";
    PrintType(op->min.dtype(), stream);
    stream << " {\n";
    int for_scope = BeginScope();
    PrintStmt(op->body);
    this->EndScope(for_scope);
    PrintIndent();
    stream << "}\n";
  } else {
    std::string upperBoundId =
        SSAGetID(PrintExpr(arith::Analyzer().Simplify(op->extent + op->min)),
                 op->extent->dtype);
    assert(op->extent.dtype().is_int() || op->extent.dtype().is_uint());
    assert(op->min.dtype() == op->extent.dtype());
    std::string vid =
        SSAGetID(AllocVarID(op->loop_var.get()), op->loop_var->dtype);
    std::string lowerBoundId = SSAGetID(PrintExpr(op->min), op->min->dtype);
    std::string extentId = SSAGetID(PrintExpr(op->extent), op->extent->dtype);
    auto stepNode = std::make_unique<IntImm>(op->min.dtype(), 1);
    auto stepId = SSAGetID(PrintExpr(*stepNode), stepNode->dtype());
    PrintIndent();
    stream << "scf.<<<invalid-for-type %" << ForKind2String(op->kind) << ">>>"
           << " %" << vid << " = %" << lowerBoundId << " to %" << upperBoundId
           << " step %" << stepId << " : ";
    PrintType(op->min.dtype(), stream);
    stream << " {\n";
    int for_scope = BeginScope();
    PrintStmt(op->body);
    this->EndScope(for_scope);
    PrintIndent();
    stream << "}\n";
  }
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const IfThenElseNode *op) {
  std::string cond = SSAGetID(PrintExpr(op->condition), op->condition->dtype);
  PrintIndent();
  stream << "scf.if %" << cond << " {\n";
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);
  if (op->else_case) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case.value());
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangCOMMONIR::PrintSSAAssign(const std::string &target,
                                             const std::string &src,
                                             DataType t) {
  PrintIndent();
  stream << "%" << target << " = " << src << "\n";
}

void CodeGenTileLangCOMMONIR::PrintShape(Array<PrimExpr> shape,
                                         std::string delimiter,
                                         std::ostream &os) {
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != 0)
      os << delimiter;
    os << shape[i];
  }
}

void CodeGenTileLangCOMMONIR::PrintType(DataType t,
                                        std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    // ICHECK(t.is_scalar()) << "do not yet support vector types";
    // os << "void*";
    return;
  }

  if (t.is_void()) {
    //    os << "void";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.is_scalar()) {
        os << "f16";
      } else {
        fail = true;
      }
      break;
    case 32:
      os << "f32";
      break;
    case 64:
      os << "f64";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "bf16";
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t == DataType::Bool()) {
    os << "i1";
    return;
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        os << "i1";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 4: {
      if (t.is_scalar()) {
        os << "i4";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 8: {
      if (t.is_scalar()) {
        os << "i8";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 16: {
      if (t.is_scalar()) {
        os << "i16";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.is_scalar()) {
        os << "i32";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.is_scalar()) {
        os << "i64";
      }
      return;
    }
    default:
      fail = true;
      break;
    }
    if (!fail) {
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t;
}

void CodeGenTileLangCOMMONIR::PrintStorageScope(const std::string &scope,
                                                std::ostream &os) { // NOLINT(*)
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const FloorDivNode *op,
                                         std::ostream &os) {
  // FIXME: The floor div in python is not the same as arith.divsi in negative
  // scenarios.
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "divsi", os);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "divf", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const FloorModNode *op,
                                         std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "remsi", os);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "remf", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const LTNode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi slt,", os);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ult,", os);
  } else {
    PrintBinary(op, "cmpf olt,", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const NENode *op, std::ostream &os) {
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ne,", os);
  } else {
    PrintBinary(op, "cmpf one,", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const EQNode *op, std::ostream &os) {
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi eq,", os);
  } else {
    PrintBinary(op, "cmpf oeq,", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const LENode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sle,", os);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ule,", os);
  } else {
    PrintBinary(op, "cmpf ole,", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const GENode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sge,", os);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi uge,", os);
  } else {
    PrintBinary(op, "cmpf oge,", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const GTNode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sgt,", os);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ugt,", os);
  } else {
    PrintBinary(op, "cmpf ogt,", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const CastNode *op, std::ostream &os) {
  bool srcIsFloat =
      op->value->dtype.is_float() || op->value->dtype.is_bfloat16();
  bool srcIsInt = op->value->dtype.is_int();
  bool srcIsUInt = op->value->dtype.is_uint();
  bool targetIsFloat = op->dtype.is_float() || op->dtype.is_bfloat16();
  bool targetIsInt = op->dtype.is_int();
  bool targetIsUInt = op->dtype.is_uint();
  auto val = PrintExpr(op->value);
  if (srcIsFloat && targetIsInt) {
    os << "arith.fptosi \%" << val << " : ";
  } else if (srcIsFloat && targetIsUInt) {
    os << "arith.fptoui \%" << val << " : ";
  } else if (srcIsInt && targetIsFloat) {
    os << "arith.sitofp \%" << val << " : ";
  } else if (srcIsUInt && targetIsFloat) {
    os << "arith.uitofp \%" << val << " : ";
  } else if (targetIsInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extsi \%" << val << " : ";
    } else {
      os << "arith.trunci \%" << val << " : ";
    }
  } else if (targetIsUInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extui \%" << val << " : ";
    } else {
      os << "arith.trunci \%" << val << " : ";
    }
  } else if (targetIsFloat) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extf \%" << val << " : ";
    } else {
      os << "arith.truncf \%" << val << " : ";
    }
  }
  PrintType(op->value->dtype, os);
  os << " to ";
  PrintType(op->dtype, os);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const BufferLoadNode *op,
                                         std::ostream &os) {

  std::ostringstream temp;
  Buffer buffer_data = op->buffer;

  DataType buffer_type = buffer_data->dtype;
  String buffer_name = buffer_data->name;
  Array<PrimExpr> buffer_shape = buffer_data->shape;
  int dim = buffer_shape.size();

  String buffer_name_val = "";
  if (auto memrefInfo = dynamic_cast<Memref *>(type_info[buffer_name])) {
    if (memrefInfo->is_arg) {
      buffer_name_val = buffer_name + "_Recast";
    } else {
      buffer_name_val = buffer_name;
    }
  } else {
    LOG(FATAL) << buffer_name << " should be a memref";
  }

  Array<String> cast_index_array = GenConvertIndex(op->indices);
  temp << "memref.load  \%" + buffer_name_val;
  temp << "[";
  for (int i = 0; i < dim; i++) {
    if (i > 0) {
      temp << ", ";
    }
    temp << cast_index_array[i];
  }
  temp << "] :";
  String data_info = GetMemrefInfo(buffer_name_val);
  temp << data_info;
  os << SSAGetID(temp.str(), buffer_type);
}

Array<String> CodeGenTileLangCOMMONIR::GenConvertIndex(Array<PrimExpr> exprs) {
  Array<String> cast_array;
  for (PrimExpr curr_expr : exprs) {
    if (auto curr_expr_int = curr_expr.as<IntImmNode>()) {
      cast_array.push_back(std::to_string(curr_expr_int->value));
    } else {
      DataType indice_type = curr_expr->dtype;
      std::ostringstream temp;
      std::string var_name;
      if (!curr_expr.as<VarNode>()) {
        var_name = SSAGetID(PrintExpr(curr_expr), indice_type);
      } else {
        var_name = PrintExpr(curr_expr);
      }
      temp << "arith.index_cast \%" << var_name << " : ";
      PrintType(indice_type, temp);
      temp << " to index";
      String cast_indice_name = "\%" + SSAGetID(temp.str(), indice_type);
      cast_array.push_back(cast_indice_name);
    }
  }
  return cast_array;
}

unsigned long ComputeOffset(Memref *src_buffer, Array<PrimExpr> op_offset) {
  if (src_buffer->var_offset)
    return -1;
  if (src_buffer->stride_int.size() != src_buffer->dim)
    return -1;
  unsigned long offset = src_buffer->offset;
  for (int i = 0; i < src_buffer->dim; i++) {
    const int64_t *op_off = as_const_int(op_offset[i]);
    if (op_off == nullptr)
      return -1;
    offset += (*op_off) * src_buffer->stride_int[i];
  }
  return offset;
}

String
CodeGenTileLangCOMMONIR::GenSubviewFromRegion(const CallNode *region_node) {
  tvm::tl::RegionOp regionop(region_node->args);
  return GenSubviewFromRegion(regionop->GetBuffer(), regionop->GetRanges());
}

String CodeGenTileLangCOMMONIR::GenSubviewFromRegion(Buffer buffer_data,
                                                     Array<Range> range) {
  std::ostringstream temp;
  DataType buffer_type = buffer_data->dtype;
  String buffer_name = buffer_data->name;
  Array<PrimExpr> buffer_shape = buffer_data->shape;
  int dim = buffer_shape.size();
  Array<PrimExpr> region_shape, region_indeces;
  for (Range r : range) {
    region_shape.push_back(r.get()->extent);
    region_indeces.push_back(r.get()->min);
  }
  String buffer_name_val = "";
  if (auto memrefInfo = dynamic_cast<Memref *>(type_info[buffer_name])) {
    if (memrefInfo->is_arg) {
      buffer_name_val = buffer_name + "_Recast";
    } else {
      buffer_name_val = buffer_name;
    }
  } else {
    LOG(FATAL) << buffer_name << " should be a memref";
  }
  auto *src_memref = dynamic_cast<Memref *>(type_info[buffer_name_val]);
  ICHECK(src_memref) << buffer_name_val << " should be a memref";

  String new_buffer_name = buffer_name_val;
  String src_data_info = GetMemrefInfo(buffer_name_val);
  if (!(IsEqual(buffer_shape, region_shape) && AllZero(region_indeces))) {
    Array<String> cast_offset_array = GenConvertIndex(region_indeces);
    Array<String> cast_shape_array = GenConvertIndex(region_shape);
    unsigned long offset = ComputeOffset(src_memref, region_indeces);
    Array<PrimExpr> result_shape = GetSubviewResultShape(region_shape);
    Array<PrimExpr> result_stride =
        GetSubviewResultStride(region_shape, src_memref->stride);
    new_buffer_name = buffer_name_val + "_subview";
    auto tempMemref = new Memref(new_buffer_name, result_shape, buffer_type,
                                 src_memref->address_space, offset == -1,
                                 result_stride, offset);
    String dst_data_info = GetMemrefInfo(tempMemref);
    temp << "memref.subview \%" + buffer_name_val;
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << cast_offset_array[i];
    }
    temp << "]";
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << cast_shape_array[i];
    }
    temp << "]";
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << "1";
    }
    temp << "]";
    temp << " : ";
    temp << src_data_info;
    temp << " to ";
    temp << dst_data_info;
    delete tempMemref;
    new_buffer_name = SSAGetID(temp.str(), buffer_type);
    this->type_info[new_buffer_name] = new Memref(
        new_buffer_name, result_shape, buffer_type, src_memref->address_space,
        offset == -1, result_stride, offset);
  }
  return new_buffer_name;
}

String CodeGenTileLangCOMMONIR::CreateMemrefToTensor(String src_data_name) {
  if (!dynamic_cast<Memref *>(type_info[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a memref";
  }
  Memref *src_memref = dynamic_cast<Memref *>(type_info[src_data_name]);
  DataType src_dtype = src_memref->dtype;
  String new_tensor_name = src_data_name + "_buffer";
  auto tempTensor = new Tensor(new_tensor_name, *src_memref);
  std::ostringstream temp;
  temp << "bufferization.to_tensor %" << src_data_name
       << " restrict writable : " << GetMemrefInfo(src_data_name);
  temp << " to " << GetTensorInfo(tempTensor);
  new_tensor_name = SSAGetID(temp.str(), src_dtype);
  tempTensor->var_id = new_tensor_name;
  this->type_info_tensor[new_tensor_name] = tempTensor;

  return new_tensor_name;
}

String CodeGenTileLangCOMMONIR::CastTensorToTensor(String src_data_name,
                                                   DataType dtype_in) {
  if (!dynamic_cast<Tensor *>(type_info_tensor[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a tensor";
  }

  Tensor *src_tensor = dynamic_cast<Tensor *>(type_info_tensor[src_data_name]);
  DataType src_dtype = src_tensor->dtype;

  if (src_dtype == dtype_in) {
    return src_data_name;
  }

  String cast_tensor_name = src_data_name + "_cast";
  auto tempTensor = new Tensor(cast_tensor_name, src_tensor->shape, dtype_in,
                               src_tensor->address_space);

  std::ostringstream temp;
  temp << GetCastOp(src_dtype, dtype_in) << " %" << src_data_name << " : ";
  temp << GetTensorInfo(src_data_name) << " to ";
  temp << GetTensorInfo(tempTensor);

  cast_tensor_name = SSAGetID(temp.str(), dtype_in);
  tempTensor->var_id = cast_tensor_name;
  this->type_info_tensor[cast_tensor_name] = tempTensor;

  return cast_tensor_name;
}

String CodeGenTileLangCOMMONIR::CreateNewTensor(String src_data_name,
                                                String input_data_name) {
  if (!dynamic_cast<Memref *>(type_info[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a memref";
  }
  String new_tensor_name = input_data_name + "_local_tensor_tmp";
  auto tempTensor = new Tensor(
      new_tensor_name, *(dynamic_cast<Memref *>(type_info[src_data_name])));
  std::ostringstream temp;
  temp << "tensor.empty() :" << GetTensorInfo(tempTensor);
  new_tensor_name = SSAGetID(temp.str(), tempTensor->dtype);
  tempTensor->var_id = new_tensor_name;
  this->type_info_tensor[new_tensor_name] = tempTensor;
  return new_tensor_name;
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const CallNode *op, std::ostream &os) {
  if (op->op.same_as(Op::Get("tl.tileop.fill"))) {
    FillCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.tileop.copy"))) {
    CopyCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.tileop.gemm")) ||
             op->op.same_as(Op::Get("tl.tileop.gemm_py"))) {
    GemmCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.infinity"))) {
    InfinityCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.tileop.reduce"))) {
    ReduceCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tir.fabs"))) {
    ICHECK_EQ(op->args.size(), 1U) << "fabs expects 1 argument";
    std::string operand = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
    if (op->dtype.is_float()) {
      os << "math.absf %" << operand << " : ";
    } else {
      os << "math.absi %" << operand << " : ";
    }
    PrintType(op->dtype, os);
  } else if (op->op.same_as(Op::Get("tir.sqrt"))) {
    ICHECK_EQ(op->args.size(), 1U) << "sqrt expects 1 argument";
    std::string operand = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
    os << "math.sqrt %" << operand << " : ";
    PrintType(op->dtype, os);
  } else if (op->op.same_as(Op::Get("tir.exp"))) {
    ICHECK_EQ(op->args.size(), 1U) << "exp expects 1 argument";
    std::string operand = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
    os << "math.exp %" << operand << " : ";
    PrintType(op->dtype, os);
  } else if (op->op.same_as(Op::Get("tir.exp2"))) {
    ICHECK_EQ(op->args.size(), 1U) << "exp2 expects 1 argument";
    std::string operand = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
    os << "math.exp2 %" << operand << " : ";
    PrintType(op->dtype, os);
  } else if (op->op.same_as(Op::Get("tir.tanh"))) {
    ICHECK_EQ(op->args.size(), 1U) << "tanh expects 1 argument";
    std::string operand = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
    os << "math.tanh %" << operand << " : ";
    PrintType(op->dtype, os);
  } else if (op->op.same_as(Op::Get("tir.log"))) {
    ICHECK_EQ(op->args.size(), 1U) << "log expects 1 argument";
    std::string operand = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
    os << "math.log %" << operand << " : ";
    PrintType(op->dtype, os);
  } else if (op->op.same_as(Op::Get("tir.rsqrt"))) {
    StubCodegen(op, os, "tir.rsqrt");
  } else if (op->op.same_as(Op::Get("tir.sigmoid"))) {
    StubCodegen(op, os, "tir.sigmoid");
  } else if (op->op.same_as(builtin::if_then_else())) {
    IfThenElseCodegen(op, os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangCOMMONIR::StubCodegen(const CallNode *op, std::ostream &os,
                                          String stubname) {
  this->PrintIndent();
  this->stream << stubname << "\n";
}

void CodeGenTileLangCOMMONIR::FillCodegen(const CallNode *op,
                                          std::ostream &os) {
  tvm::tl::Fill fillop(op->args);
  std::string value_name =
      SSAGetID(PrintExpr(fillop->value), fillop->value->dtype);

  this->PrintIndent();
  this->stream << "linalg.fill ins(%" << value_name << " : ";
  PrintType(fillop->value->dtype, this->stream);
  this->stream << ") outs(%" << fillop->dst.get()->name << " : ";
  this->stream << GetMemrefInfo(fillop->dst.get()->name) << ")\n";
}

void CodeGenTileLangCOMMONIR::CopyCodegen(const CallNode *op,
                                          std::ostream &os) {
  tvm::tl::Copy copyop(op->args);

  String src_data_name = GenSubviewFromRegion(copyop->src, copyop->src_range);
  String dst_data_name = GenSubviewFromRegion(copyop->dst, copyop->dst_range);

  if (!dynamic_cast<Memref *>(type_info[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a memref";
  }
  if (!dynamic_cast<Memref *>(type_info[dst_data_name])) {
    LOG(FATAL) << dst_data_name << " should be a memref";
  }

  DataType src_dtype = dynamic_cast<Memref *>(type_info[src_data_name])->dtype;
  DataType dst_dtype = dynamic_cast<Memref *>(type_info[dst_data_name])->dtype;
  if (src_dtype == dst_dtype) {
    this->PrintIndent();
    this->stream << "memref.copy"
                 << " \%" << src_data_name << ", "
                 << "\%" << dst_data_name << " : ";
    this->stream << GetMemrefInfo(src_data_name) << " to "
                 << GetMemrefInfo(dst_data_name) << "\n";
  } else {
    LOG(INFO) << "memref.copy: src_dtype: " << src_dtype
              << " != dst_dtype: " << dst_dtype;

    std::ostringstream temp;

    String new_tensor_name = CreateMemrefToTensor(src_data_name);
    String cast_tensor_name = CastTensorToTensor(new_tensor_name, dst_dtype);

    this->PrintIndent();
    this->stream << "bufferization.materialize_in_destination \%";
    this->stream << cast_tensor_name << " in writable  \%" << dst_data_name
                 << " : (";
    this->stream << GetTensorInfo(cast_tensor_name) << " , "
                 << GetMemrefInfo(dst_data_name) << ") -> ()";
    this->stream << "\n";
  }
}

void CodeGenTileLangCOMMONIR::IfThenElseCodegen(const CallNode *op,
                                                std::ostream &os) {
  // args[0] is the condition (bool/i1), must use its own dtype, not op->dtype
  std::string cond = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
  // Ensure true/false values are proper SSA IDs (handles inline constant exprs)
  std::string true_val = SSAGetID(PrintExpr(op->args[1]), op->args[1]->dtype);
  std::string false_val = SSAGetID(PrintExpr(op->args[2]), op->args[2]->dtype);
  std::ostringstream temp;
  temp << "arith.select %" << cond << ", %" << true_val << ", %" << false_val
       << " : ";
  PrintType(op->dtype, temp);
  std::string result = SSAGetID(temp.str(), op->dtype);
  os << result;
}

void CodeGenTileLangCOMMONIR::GemmCodegen(const CallNode *op,
                                          std::ostream &os) {
  tvm::tl::Gemm gemmop(op->args);
  // todo(dkx): support clearAccum_ = True
  ICHECK(is_zero(gemmop->clearAccum_))
      << "Currently we only support: clearAccum_ must be const_false";
  // todo(dkx): maybe not necessary
  // ICHECK(gemmop->policy_ == tvm::tl::GemmWarpPolicyType::kSquare)
  //     << "Currently we only support: policy_ must be kSquare";
  ICHECK(gemmop->kPack_ == 1) << "Currently we only support: kPack_ must be 1";
  ICHECK(gemmop->wgWait_ == 0)
      << "Currently we only support: wgWait_ must be 0";

  Buffer a_buffer = gemmop->a_;
  Buffer b_buffer = gemmop->b_;
  Buffer c_buffer = gemmop->c_;
  String a_buffer_name = a_buffer->name;
  String b_buffer_name = b_buffer->name;
  String c_buffer_name = c_buffer->name;
  String A_shared_tensor = CreateMemrefToTensor(a_buffer_name);
  String B_shared_tensor = CreateMemrefToTensor(b_buffer_name);
  String C_shared_tensor = CreateMemrefToTensor(c_buffer_name);
  String new_tensor_name = CreateNewTensor(c_buffer_name, C_shared_tensor);
  std::ostringstream temp;
  DataType dst_dtype = this->type_info_tensor[new_tensor_name]->dtype;

  auto transpose_if_needed = [this,
                              &temp](const String &tensor_name) -> String {
    ICHECK(this->type_info_tensor.count(tensor_name))
        << "Can not find tensor ssa object: " << tensor_name;
    auto *src_tensor =
        dynamic_cast<Tensor *>(this->type_info_tensor[tensor_name]);
    ICHECK(src_tensor) << tensor_name << " should be a tensor";
    ICHECK_EQ(src_tensor->shape.size(), 2)
        << "Only support 2D transpose for gemm operands";

    Array<PrimExpr> transposed_shape;
    transposed_shape.push_back(src_tensor->shape[1]);
    transposed_shape.push_back(src_tensor->shape[0]);

    String transposed_tensor_name = tensor_name + "_transposed";
    auto *transposed_tensor =
        new Tensor(transposed_tensor_name, transposed_shape, src_tensor->dtype,
                   src_tensor->address_space);

    temp.str("");
    temp.clear();
    temp << "tensor.empty() :" << GetTensorInfo(transposed_tensor);
    String transposed_init = SSAGetID(temp.str(), src_tensor->dtype);
    transposed_tensor->var_id = transposed_init;
    this->type_info_tensor[transposed_init] = transposed_tensor;

    temp.str("");
    temp.clear();
    temp << "linalg.transpose ins(%" << tensor_name << " : "
         << GetTensorInfo(tensor_name) << ") outs(%" << transposed_init << " : "
         << GetTensorInfo(transposed_init) << ") permutation = [1, 0]";
    String transposed_out = SSAGetID(temp.str(), src_tensor->dtype);
    auto *transposed_out_tensor =
        new Tensor(transposed_out, transposed_shape, src_tensor->dtype,
                   src_tensor->address_space);
    transposed_out_tensor->var_id = transposed_out;
    this->type_info_tensor[transposed_out] = transposed_out_tensor;
    return transposed_out;
  };

  String matmul_a_tensor =
      gemmop->transA_ ? transpose_if_needed(A_shared_tensor) : A_shared_tensor;
  String matmul_b_tensor =
      gemmop->transB_ ? transpose_if_needed(B_shared_tensor) : B_shared_tensor;

  temp.str("");
  temp.clear();
  temp << "linalg.matmul ins(\%" << matmul_a_tensor << ", \%" << matmul_b_tensor
       << " : " << GetTensorInfo(matmul_a_tensor) << ", "
       << GetTensorInfo(matmul_b_tensor) << ") ";
  temp << "outs(\%" << new_tensor_name << " : "
       << GetTensorInfo(new_tensor_name) << ") -> "
       << GetTensorInfo(new_tensor_name);

  String C_tensor_out = SSAGetID(temp.str(), dst_dtype);
  temp.str("");
  temp.clear();
  if (dst_dtype.is_int() || dst_dtype.is_uint()) {
    temp << "arith.addi ";
  } else if (dst_dtype.is_float()) {
    temp << "arith.addf ";
  }
  temp << "\%" << C_shared_tensor << ", \%" << C_tensor_out << " : "
       << GetTensorInfo(C_shared_tensor);
  String tmp_out = SSAGetID(temp.str(), dst_dtype);
  this->PrintIndent();
  this->stream << "bufferization.materialize_in_destination %" << tmp_out
               << " in writable %" << c_buffer_name << " : ("
               << GetTensorInfo(new_tensor_name) << ", "
               << GetMemrefInfo(c_buffer_name) << ") -> ()\n";
}

void CodeGenTileLangCOMMONIR::InfinityCodegen(const CallNode *op,
                                              std::ostream &os) {
  const DataType &dtype = op->dtype;
  ICHECK_EQ(dtype.lanes(), 1);
  if (dtype.is_float()) {
    if (dtype.bits() == 64 || dtype.bits() == 32 || dtype.bits() == 16) {
      PrimExpr imm_exp =
          FloatImm(dtype, std::numeric_limits<float>::infinity(), op->span);
      os << SSAGetID(PrintExpr(imm_exp), dtype);
      return;
    }
  } else if (dtype.is_bfloat16()) {
    PrimExpr imm_exp =
        FloatImm(dtype, std::numeric_limits<float>::infinity(), op->span);
    os << SSAGetID(PrintExpr(imm_exp), dtype);
    return;
  }
  LOG(FATAL) << "Cannot decide infinity for type " << dtype;
  throw;
}

void CodeGenTileLangCOMMONIR::ReduceCodegen(const CallNode *op,
                                            std::ostream &os) {
  tvm::tl::ReduceOp reduceop(op->args);

  ICHECK(reduceop->type->isSum() || reduceop->type->isMax())
      << "Currently we only support: sum or max";

  String src_data_name =
      GenSubviewFromRegion(reduceop->src, reduceop->srcRegion_->region);
  String dst_data_name =
      GenSubviewFromRegion(reduceop->dst, reduceop->dstRegion_->region);

  auto *src_memref = dynamic_cast<Memref *>(type_info[src_data_name]);
  auto *dst_memref = dynamic_cast<Memref *>(type_info[dst_data_name]);
  ICHECK(src_memref) << src_data_name << " should be a memref";
  ICHECK(dst_memref) << dst_data_name << " should be a memref";

  DataType src_dtype = src_memref->dtype;
  DataType dst_dtype = dst_memref->dtype;

  String src_tensor_name = CreateMemrefToTensor(src_data_name);
  // Always use a temporary tensor as reduce outs, then materialize to memref.
  // This keeps reduction in tensor form instead of memref-backed tensor views.
  String init_tensor_name = CreateNewTensor(dst_data_name, "max_vals");

  if (reduceop->clear) {
    PrimExpr init_value;
    if (reduceop->type->isSum()) {
      init_value = make_zero(dst_dtype);
    } else if (dst_dtype.is_int()) {
      init_value = make_const(dst_dtype, -(1LL << (dst_dtype.bits() - 1)));
    } else if (dst_dtype.is_uint()) {
      init_value = make_const(dst_dtype, 0);
    } else if (dst_dtype.is_float() || dst_dtype.is_bfloat16()) {
      init_value = make_const(dst_dtype, -INFINITY);
    } else {
      LOG(FATAL) << "Unsupported dtype for max reduce init: " << dst_dtype;
    }
    std::string init_value_name = SSAGetID(PrintExpr(init_value), dst_dtype);
    std::ostringstream fill_temp;
    fill_temp << "linalg.fill ins(%" << init_value_name << " : ";
    PrintType(dst_dtype, fill_temp);
    fill_temp << ") outs(%" << init_tensor_name << " : "
              << GetTensorInfo(init_tensor_name) << ") -> "
              << GetTensorInfo(init_tensor_name);
    String filled_from_tensor_name = init_tensor_name;
    init_tensor_name = SSAGetID(fill_temp.str(), dst_dtype);
    auto *filled_tensor_template =
        dynamic_cast<Tensor *>(type_info_tensor[filled_from_tensor_name]);
    ICHECK(filled_tensor_template)
        << filled_from_tensor_name << " should be a tensor";
    auto *filled_tensor =
        new Tensor(init_tensor_name, filled_tensor_template->shape, dst_dtype,
                   filled_tensor_template->address_space);
    filled_tensor->var_id = init_tensor_name;
    this->type_info_tensor[init_tensor_name] = filled_tensor;
  }

  std::ostringstream reduce_temp;
  reduce_temp << "linalg.reduce ins(%" << src_tensor_name << " : "
              << GetTensorInfo(src_tensor_name) << ") outs(%"
              << init_tensor_name << " : " << GetTensorInfo(init_tensor_name)
              << ") dimensions = [" << reduceop->dim << "]\n";
  reduce_temp << "  (%in: ";
  PrintType(src_dtype, reduce_temp);
  reduce_temp << ", %acc: ";
  PrintType(dst_dtype, reduce_temp);
  reduce_temp << ") {\n";

  std::string rhs_name = "in";
  if (src_dtype != dst_dtype) {
    reduce_temp << "    %in_cast = " << GetCastOp(src_dtype, dst_dtype)
                << " %in : ";
    PrintType(src_dtype, reduce_temp);
    reduce_temp << " to ";
    PrintType(dst_dtype, reduce_temp);
    reduce_temp << "\n";
    rhs_name = "in_cast";
  }

  reduce_temp << "    %reduced = ";
  if (reduceop->type->isSum()) {
    if (dst_dtype.is_int() || dst_dtype.is_uint()) {
      reduce_temp << "arith.addi %acc, %" << rhs_name << " : ";
    } else if (dst_dtype.is_float() || dst_dtype.is_bfloat16()) {
      reduce_temp << "arith.addf %acc, %" << rhs_name << " : ";
    } else {
      LOG(FATAL) << "Unsupported dtype for sum reduce: " << dst_dtype;
    }
  } else if (dst_dtype.is_int()) {
    reduce_temp << "arith.maxsi %acc, %" << rhs_name << " : ";
  } else if (dst_dtype.is_uint()) {
    reduce_temp << "arith.maxui %acc, %" << rhs_name << " : ";
  } else if (dst_dtype.is_float() || dst_dtype.is_bfloat16()) {
    reduce_temp << "arith.maxnumf %acc, %" << rhs_name << " : ";
  } else {
    LOG(FATAL) << "Unsupported dtype for max reduce: " << dst_dtype;
  }
  PrintType(dst_dtype, reduce_temp);
  reduce_temp << "\n";
  reduce_temp << "    linalg.yield %reduced : ";
  PrintType(dst_dtype, reduce_temp);
  reduce_temp << "\n";
  reduce_temp << "  }";

  String reduced_tensor_name = SSAGetID(reduce_temp.str(), dst_dtype);
  auto *reduced_tensor_template =
      dynamic_cast<Tensor *>(type_info_tensor[init_tensor_name]);
  ICHECK(reduced_tensor_template) << init_tensor_name << " should be a tensor";
  auto *reduced_tensor =
      new Tensor(reduced_tensor_name, reduced_tensor_template->shape, dst_dtype,
                 reduced_tensor_template->address_space);
  reduced_tensor->var_id = reduced_tensor_name;
  this->type_info_tensor[reduced_tensor_name] = reduced_tensor;

  this->PrintIndent();
  this->stream << "bufferization.materialize_in_destination %"
               << reduced_tensor_name << " in writable %" << dst_data_name
               << " : (" << GetTensorInfo(reduced_tensor_name) << ", "
               << GetMemrefInfo(dst_data_name) << ") -> ()\n";
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const LetStmtNode *op) {
  std::string value = PrintExpr(op->value);
  PrintIndent();
  this->stream << '%' << AllocVarID(op->var.get()) << " = " << value << "\n";
  PrintStmt(op->body);
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const BufferStoreNode *op) {
  std::string value = SSAGetID(PrintExpr(op->value), op->value->dtype);

  Buffer buffer_data = op->buffer;
  DataType buffer_type = buffer_data->dtype;
  String buffer_name = buffer_data->name;
  Array<PrimExpr> buffer_shape = buffer_data->shape;
  int dim = buffer_shape.size();

  String buffer_name_val = "";
  if (auto memrefInfo = dynamic_cast<Memref *>(type_info[buffer_name])) {
    if (memrefInfo->is_arg) {
      buffer_name_val = buffer_name + "_Recast";
    } else {
      buffer_name_val = buffer_name;
    }
  } else {
    LOG(FATAL) << buffer_name << " should be a memref";
  }

  Array<String> cast_index_array = GenConvertIndex(op->indices);
  PrintIndent();
  this->stream << "memref.store  \%" + value + ", \%" + buffer_name_val;
  this->stream << "[";
  for (int i = 0; i < dim; i++) {
    if (i > 0) {
      this->stream << ", ";
    }
    this->stream << cast_index_array[i];
  }
  this->stream << "] :";

  String data_info = GetMemrefInfo(buffer_name_val);
  this->stream << data_info << "\n";
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "thread_extent") {
    IterVar iv = Downcast<IterVar>(op->node);
    if ((iv->thread_tag == "blockIdx.x" || iv->thread_tag == "blockIdx.y" ||
         iv->thread_tag == "blockIdx.z") &&
        iv->var->name_hint != "_") {
      int arg_index = -1;
      if (iv->thread_tag == "blockIdx.x") {
        arg_index = 3;
      } else if (iv->thread_tag == "blockIdx.y") {
        arg_index = 4;
      } else if (iv->thread_tag == "blockIdx.z") {
        arg_index = 5;
      }
      std::ostringstream temp;
      temp << "arith.constant 0"
           << " : ";
      PrintType(iv->var->dtype, temp);
      std::string vid = SSAGetID(temp.str(), iv->var->dtype);
      auto block_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "%" << block_id_ << " = arith.addi  %" << vid << ", "
                   << this->thread_context_args[arg_index] << ": i32\n";
    } else if ((iv->thread_tag == "threadIdx.x" ||
                iv->thread_tag == "threadIdx.y" ||
                iv->thread_tag == "threadIdx.z") &&
               iv->var->name_hint != "_") {
      // todo(dkx): should handle this dilemma on npu
      auto block_id_ = AllocVarID(iv->var.get());
    }
    this->VisitStmt(op->body);
    return;
  }

  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string scope = GetPtrStorageScope(op->buffer_var);

  std::string vid = AllocVarID(op->buffer_var.get());
  String address_space = GetAddressSpace(scope);
  if (!op->buffer_var->type_annotation.defined()) {
    LOG(FATAL) << "AllocateNode buffer_var must have a type annotation";
  }
  auto ptr_type = op->buffer_var->type_annotation.as<PointerTypeNode>();
  if (!ptr_type) {
    LOG(FATAL) << "AllocateNode buffer_var must be a pointer type";
  }
  auto prim_type = ptr_type->element_type.as<PrimTypeNode>();
  if (!prim_type) {
    LOG(FATAL) << "AllocateNode buffer_var must point to a primitive type";
  }
  Buffer buffer = decl_buffer(op->extents, prim_type->dtype, vid, scope,
                              Array<IntImm>(), Span());
  vmap.Set(op->buffer_var, buffer);

  this->type_info[vid] =
      new Memref(vid, op->extents, op->dtype, address_space, false);
  this->PrintIndent();
  stream << "%" << vid << " = "
         << "memref.alloc() : " << GetMemrefInfo(vid) << "\n";

  this->VisitStmt(op->body);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const MinNode *op, std::ostream &os) {
  if (op->dtype.is_int()) {
    PrintBinary(op, "minsi", os);
  } else if (op->dtype.is_uint()) {
    PrintBinary(op, "minui", os);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "minnumf", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const MaxNode *op, std::ostream &os) {
  if (op->dtype.is_int()) {
    PrintBinary(op, "maxsi", os);
  } else if (op->dtype.is_uint()) {
    PrintBinary(op, "maxui", os);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "maxnumf", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const AddNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "addi", os);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "addf", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const SubNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "subi", os);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "subf", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const FloatImmNode *op,
                                         std::ostream &os) { // NOLINT(*)
  std::ostringstream temp;
  if (op->value == -std::numeric_limits<float>::infinity()) {
    temp << "arith.constant 0xFF800000 : ";
  } else if (op->value == std::numeric_limits<float>::infinity()) {
    temp << "arith.constant 0x7F800000 : ";
  } else {
    temp << "arith.constant ";
    double val = op->value;
    if (std::floor(val) == val && std::isfinite(val)) {
      temp << static_cast<long long>(val) << ".0";
    } else {
      temp << val;
    }
    temp << " : ";
  }
  PrintType(op->dtype, temp);
  os << SSAGetID(temp.str(), op->dtype);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const IntImmNode *op,
                                         std::ostream &os) {
  std::ostringstream temp;
  temp << "arith.constant ";
  if (op->dtype.is_bool()) {
    temp << (op->value == 1 ? "true" : "false");
  } else {
    temp << op->value << " : ";
    PrintType(op->dtype, temp);
  }
  os << SSAGetID(temp.str(), op->dtype);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const MulNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "muli", os);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "mulf", os);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const AndNode *op, std::ostream &os) {
  assert(op->a.dtype().is_int() || op->a.dtype().is_uint());
  assert(op->b.dtype().is_int() || op->b.dtype().is_uint());
  PrintBinary(op, "andi", os);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const OrNode *op, std::ostream &os) {
  assert(op->a.dtype().is_int() || op->a.dtype().is_uint());
  assert(op->b.dtype().is_int() || op->b.dtype().is_uint());
  PrintBinary(op, "ori", os);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const DivNode *op, std::ostream &os) {
  if (op->dtype.is_int()) {
    PrintBinary(op, "divsi", os);
  } else if (op->dtype.is_uint()) {
    PrintBinary(op, "divui", os);
  } else if (op->dtype.is_float() || op->dtype.is_bfloat16()) {
    PrintBinary(op, "divf", os);
  } else {
    LOG(FATAL) << "Unsupported dtype for div: " << op->dtype;
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const SelectNode *op,
                                         std::ostream &os) {
  auto PrintOp = [this](const PrimExpr &operand) {
    std::ostringstream tmpos;
    if (operand.as<tvm::tir::IntImmNode>() ||
        operand.as<tvm::tir::FloatImmNode>() ||
        operand.as<tvm::tir::VarNode>()) {
      PrintExpr(operand, tmpos << "%");
    } else {
      std::string op_name = SSAGetID(PrintExpr(operand), operand->dtype);
      tmpos << "%" << op_name;
    }
    return tmpos.str();
  };

  os << "arith.select " << PrintOp(op->condition) << ", "
     << PrintOp(op->true_value) << ", " << PrintOp(op->false_value) << " : ";
  PrintType(op->dtype, os);
}

void PrintHostFunc(const PrimFunc &f, const std::string &name, std::ostream &os,
                   int core) {
  os << "extern \"C\" void call(";
  std::vector<std::string> arg_names;
  for (size_t i = 0; i < f->params.size(); ++i) {
    auto v = f->params[i];
    if (i != 0) {
      os << ", ";
    }
    arg_names.push_back(v->name_hint);
    os << "uint8_t* " << v->name_hint;
  }
  os << ", aclrtStream stream) {\n  ";

  os << name << "<<<" << core << ", nullptr, stream>>>(";
  for (auto &arg_name : arg_names) {
    os << arg_name;
    if (arg_name != arg_names.back()) {
      os << ", ";
    }
  }
  os << ");\n";
  os << "}\n";
}

void CodeGenTileLangCOMMONIR::GenRecastFromArg(Buffer curr_buffer,
                                               String arg_name,
                                               String &recast_inst) {
  std::ostringstream res;
  String target_strides = GetBufferStrides(curr_buffer);
  String cast_name = arg_name + "_Recast";
  this->type_info[cast_name] = new Memref(cast_name, curr_buffer);
  res << "\%" << cast_name << " = ";
  res << "memref.reinterpret_cast \%";
  res << arg_name;
  res << " to offset: [0], sizes: [";
  PrintShape(curr_buffer->shape, ", ", res);
  res << "], strides: ";
  res << target_strides;
  res << " : ";
  res << GetMemrefInfo(arg_name);
  res << " to ";
  res << GetMemrefInfo(cast_name);
  res << "\n";
  recast_inst = res.str();
}

void CodeGenTileLangCOMMONIR::AddFunction(const GlobalVar &gvar,
                                          const PrimFunc &f) {
  this->stream << "module attributes {dicp.backend = \"ascend\"} {\n";

  // If the function has already been forward-declared, this is a
  // no-op.
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);

  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.has_value())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  int func_scope = this->BeginScope();
  this->PrintIndent();
  auto func_name = static_cast<std::string>(global_symbol.value());

  this->stream << "func.func @" << func_name << "(";

  std::vector<String> recast_need_insert;

  this->type_info.clear();
  size_t n = f->params.size();
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());

    if (i != 0)
      stream << ", ";

    if (v.dtype().is_handle()) {
      this->vmap = f->buffer_map;
      auto real_v = f->buffer_map[v]->data;
      String arg_name = AllocVarID(real_v.get());
      Memref *buffer = new Memref(arg_name, f->buffer_map[v], true);
      this->type_info[arg_name] = buffer;
      stream << "%" << arg_name << ": " << GetMemrefInfo(arg_name);
      String recast_inst = "";
      GenRecastFromArg(f->buffer_map[v], arg_name, recast_inst);
      recast_need_insert.push_back(recast_inst);

      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
    } else {
      stream << "%" << vid << ": ";
      CodeGenC::PrintType(GetType(v), stream);
    }
  }

  for (size_t i = 0; i < 6; ++i) {
    this->thread_context_args[i] = "\%args" + std::to_string(n + i);
    stream << ", ";
    stream << thread_context_args[i] << ": i32";
  }
  stream << ")\n";
  this->PrintIndent();
  stream << "{\n";
  int func_body_scope = this->BeginScope();
  for (String recast_inst : recast_need_insert) {
    this->PrintIndent();
    stream << recast_inst;
  }
  this->PrintStmt(f->body);
  this->EndScope(func_body_scope);
  this->PrintIndent();
  this->stream << "return\n";
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

String CodeGenTileLangCOMMONIR::GetMemrefInfo(String name) {
  if (this->type_info.count(name) == 0)
    LOG(FATAL) << "Can not find memref ssa object: " << name;
  if (!dynamic_cast<Memref *>(type_info[name])) {
    LOG(FATAL) << name << " should be a memref";
  }
  Memref *MemrefObj = dynamic_cast<Memref *>(this->type_info[name]);
  return GetMemrefInfo(MemrefObj);
}

String CodeGenTileLangCOMMONIR::GetMemrefInfo(Memref *memrefObj) {
  if (memrefObj->type_str != "")
    return memrefObj->type_str;
  std::ostringstream memref_type;
  memref_type << "memref<";
  if (memrefObj->is_arg) {
    memref_type << "?x";
  } else {
    for (PrimExpr s : memrefObj->shape) {
      if (auto s_int = as_const_int(s)) {
        memref_type << std::to_string(*s_int) + "x";
      } else {
        // not support ssa var in memref size
        memref_type << "?x";
      }
    }
  }
  PrintType(memrefObj->dtype, memref_type);
  if (!memrefObj->is_arg) {
    memref_type << ", strided<[";
    for (int i = 0; i < memrefObj->dim; i++) {
      if (i > 0)
        memref_type << ", ";
      if (memrefObj->stride.size() > 0) {
        if (auto s_int = as_const_int(memrefObj->stride[i])) {
          memref_type << std::to_string(*s_int);
        } else {
          // not support ssa var in memref size
          memref_type << "?";
        }
      } else {
        memref_type << memrefObj->stride_int[i];
      }
    }
    memref_type << "], offset:";
    if (memrefObj->var_offset)
      memref_type << "?";
    else
      memref_type << memrefObj->offset;
    memref_type << ">";
  }
  memref_type << ">";
  // memref_type << ", #address_space<" << memrefObj->address_space << ">>";
  memrefObj->type_str = memref_type.str();
  return memrefObj->type_str;
}

void Memref::GetIntStride() {
  if (stride.empty()) {
    stride_int = GetStrideFromShape(shape);
    for (unsigned long s : stride_int) {
      stride.push_back(IntImm(DataType::Int(64), s));
    }
  } else {
    for (PrimExpr s : stride) {
      if (auto s_int = as_const_int(s))
        stride_int.push_back(*s_int);
    }
  }
}

Memref::Memref(String name, Array<PrimExpr> shape_in, DataType dtype_in,
               String addr_space_in, bool var_offset_in,
               Array<PrimExpr> stride_in, int offset_in, bool is_arg_in) {
  var_id = name;
  shape = shape_in;
  stride = stride_in;
  dtype = dtype_in;
  offset = offset_in;
  is_arg = is_arg_in;
  address_space = addr_space_in;
  var_offset = var_offset_in;
  dim = shape_in.size();
  GetIntStride();
}

Memref::Memref(String name, Buffer buffer, bool is_arg_in) {
  var_id = name;
  shape = buffer->shape;
  stride = buffer->strides;
  dtype = buffer->dtype;
  offset = 0;
  is_arg = is_arg_in;
  String scope = GetPtrStorageScope(buffer->data);
  address_space = GetAddressSpace(scope);
  var_offset = false;
  dim = shape.size();
  GetIntStride();
}

String CodeGenTileLangCOMMONIR::GetTensorInfo(String name) {
  if (this->type_info_tensor.count(name) == 0)
    LOG(FATAL) << "Can not find tensor ssa object: " << name;
  if (!dynamic_cast<Tensor *>(type_info_tensor[name])) {
    LOG(FATAL) << name << " should be a tensor";
  }
  Tensor *TensorObj = dynamic_cast<Tensor *>(this->type_info_tensor[name]);
  return GetTensorInfo(TensorObj);
}

String CodeGenTileLangCOMMONIR::GetTensorInfo(Tensor *tensorObj) {
  if (tensorObj->type_str != "")
    return tensorObj->type_str;
  std::ostringstream tensor_type;
  tensor_type << "tensor<";
  for (PrimExpr s : tensorObj->shape) {
    if (auto s_int = as_const_int(s)) {
      tensor_type << std::to_string(*s_int) + "x";
    } else {
      // not support ssa var in memref size
      tensor_type << "?x";
    }
  }
  PrintType(tensorObj->dtype, tensor_type);
  tensor_type << ">";
  // tensor_type << ", #address_space<" << tensorObj->address_space << ">>";
  tensorObj->type_str = tensor_type.str();
  return tensorObj->type_str;
}

Tensor::Tensor(String name, Array<PrimExpr> shape_in, DataType dtype_in,
               String addr_space_in) {
  var_id = name;
  shape = shape_in;
  dtype = dtype_in;
  address_space = addr_space_in;
  dim = shape_in.size();
}

Tensor::Tensor(String name, Buffer buffer) {
  var_id = name;
  shape = buffer->shape;
  dtype = buffer->dtype;
  String scope = GetPtrStorageScope(buffer->data);
  address_space = GetAddressSpace(scope);
  dim = shape.size();
}
Tensor::Tensor(String name, Memref memref) {
  var_id = name;
  shape = memref.shape;
  dtype = memref.dtype;
  address_space = memref.address_space;
  dim = shape.size();
}
} // namespace codegen

} // namespace tvm
