#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <utility>
#include <vector>

auto opPrint(mlir::Operation *moduleOp) {
  auto pflags = mlir::OpPrintingFlags();
  pflags.enableDebugInfo(true, true);
  moduleOp->print(llvm::outs(), pflags);
  llvm::outs().flush();
}

auto mkLoc(mlir::MLIRContext *ctx, int line) {
  auto fileName = llvm::StringRef(__FILE__);
  auto fileLineColLoc = mlir::FileLineColLoc::get(ctx, fileName, line, 1);
  return mlir::Location(fileLineColLoc);
}

auto createMesh(
    mlir::Builder *builder, llvm::StringRef meshName,
    mlir::ArrayRef<std::pair<llvm::StringRef, int64_t>> axisNameAndSizes,
    mlir::ArrayRef<int64_t> deviceIds = {}) {
  using namespace mlir::sdy;
  auto context = builder->getContext();
  mlir::SmallVector<MeshAxisAttr> meshAxisAttrs;
  meshAxisAttrs.reserve(axisNameAndSizes.size());
  for (auto [axisName, axisSize] : axisNameAndSizes) {
    meshAxisAttrs.push_back(MeshAxisAttr::get(context, axisName, axisSize));
  }
  auto meshAttr = MeshAttr::get(context, meshAxisAttrs, deviceIds);
  // auto loc = mkLoc(context, __LINE__);
  // return builder->create<mlir::sdy::MeshOp>(loc, meshName, meshAttr);
  return meshAttr;
}

mlir::sdy::DimensionShardingAttr
createDimSharding(mlir::Builder *builder,
                  mlir::ArrayRef<mlir::sdy::AxisRefAttr> axes,
                  bool isClosed = false) {
  return mlir::sdy::DimensionShardingAttr::get(builder->getContext(), axes,
                                               isClosed);
}

mlir::sdy::AxisRefAttr createAxis(mlir::Builder *builder,
                                  llvm::StringRef name) {
  return mlir::sdy::AxisRefAttr::get(builder->getContext(), name);
}

mlir::sdy::TensorShardingAttr createTensorSharding(
    mlir::Builder *builder, llvm::StringRef meshName,
    mlir::ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings,
    mlir::ArrayRef<mlir::sdy::AxisRefAttr> replicatedAxes = {}) {
  auto context = builder->getContext();
  return mlir::sdy::TensorShardingAttr::get(context, meshName, dimShardings,
                                            replicatedAxes);
}

auto toBF16(llvm::APFloat &value) {
  bool ignored;
  value.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &ignored);
}

auto mkTensor(mlir::ArrayRef<int64_t> shape, mlir::Type ty) {
  auto tty = mlir::RankedTensorType::get(shape, ty);
  // auto size = tty.getNumElements();
  // std::vector<llvm::APFloat> buffer;
  // float data[3] = {0.3f, 0.6f, 0.9f};
  // for (auto i = 0; i < size; i++) {
  //     auto value = (data[i % 3]);
  //     buffer.push_back(llvm::APFloat(value));
  //     toBF16(buffer.back());
  // }
  // return std::pair(tty, buffer);
  return tty;
}

auto matmulFunc(
    mlir::Builder *builder, llvm::StringRef meshName,
    std::array<mlir::RankedTensorType, 2>
        ins,
    mlir::sdy::MeshAttr mesh) {
  auto ctx = builder->getContext();
  auto location = mkLoc(ctx, __LINE__);
  auto sharding = createTensorSharding(builder, meshName, {});
  std::vector<mlir::Type> itypes;
  itypes.reserve(ins.size());
  std::vector<mlir::DictionaryAttr> argAttrs;
  argAttrs.reserve(ins.size());
  auto i = 0;
  for (auto tty : ins) {
    auto shardedTensor = sharding.getLocalTensorType(tty, mesh);
    // shardedTensor.getShape()
    // auto dea = mlir::DenseElementsAttr::get(, ten);
    std::stringstream ss;
    ss << "arg" << i;
    argAttrs.push_back(builder->getDictionaryAttr(builder->getNamedAttr(ss.str(), sharding)));
    itypes.push_back(shardedTensor);
    i += 1;
  }

  auto type = builder->getFunctionType(mlir::ArrayRef(itypes), {});
  auto funcOp = mlir::func::FuncOp::create(location, "matmul", type);
  funcOp.setAllArgAttrs(argAttrs);
  return funcOp;
}

int main(int argc, char **argv) {
  const auto meshName = "mesh";
  auto daxis = std::pair(llvm::StringRef("x"), (int64_t)4);
  auto taxis = std::pair(llvm::StringRef("y"), (int64_t)2);
  auto namedAxes = {daxis, taxis};
  const auto devices =
      mlir::ArrayRef<std::pair<llvm::StringRef, int64_t>>(namedAxes);
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  mlir::sdy::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  auto threadingEnabled = mlir::MLIRContext::Threading::DISABLED;
  auto ctx = mlir::MLIRContext(registry, threadingEnabled);
  ctx.loadAllAvailableDialects();
  auto builder = mlir::OpBuilder(&ctx);
  auto moduleOp = mlir::ModuleOp::create(mkLoc(&ctx, __LINE__), "main");
  auto mesh = createMesh(&builder, meshName, devices);
  auto ty = builder.getBF16Type();
  auto in1 = mkTensor({4, 8}, ty);
  auto in2 = mkTensor({8, 12}, ty);
  moduleOp.getBody()->push_back(
      matmulFunc(&builder, meshName, {in1, in2}, mesh));
  opPrint(moduleOp);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SDY Builder", registry));
}
