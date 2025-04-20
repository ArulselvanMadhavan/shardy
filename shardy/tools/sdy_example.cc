#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

auto opPrint(mlir::Operation *moduleOp) {
  auto pflags = mlir::OpPrintingFlags();
  pflags.enableDebugInfo(true, false);
  moduleOp->print(llvm::outs(), pflags);
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

auto matmulFunc(mlir::Builder *builder, llvm::StringRef meshName) {
  auto ctx = builder->getContext();
  auto location = mkLoc(ctx, __LINE__);
  auto func_name = "matmul";
  auto sym_name = builder->getStringAttr(func_name);
  auto sym_attr = builder->getNamedAttr(llvm::StringRef("sym_name"), sym_name);
  auto globalType = mlir::RankedTensorType::get({8, 4}, builder->getF16Type());
  auto t1arg = createTensorSharding(builder, meshName, {});
  auto type = builder->getFunctionType({globalType}, {});
  auto arg_attrs = builder->getNamedAttr("arg_attrs", {t1arg});
  auto attrs1 = {sym_attr, arg_attrs};
  auto attrs = mlir::ArrayRef<mlir::NamedAttribute>(attrs1);
  auto funcOp = mlir::func::FuncOp::create(location, func_name, type, attrs);
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
  moduleOp.getBody()->push_back(matmulFunc(&builder, meshName));
  opPrint(moduleOp);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SDY Builder", registry));
}
