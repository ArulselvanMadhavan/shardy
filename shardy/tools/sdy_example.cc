#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shardy/dialect/sdy/ir/register.h"
#include <vector>

auto opPrint(mlir::Operation *moduleOp) {
  auto pflags = mlir::OpPrintingFlags();
  pflags.enableDebugInfo(true, false);
  moduleOp->print(llvm::outs(), pflags);
}

auto mkLoc(mlir::MLIRContext *mlirCtx, int line) {
  auto fileName = llvm::StringRef(__FILE__);
  auto fileLineColLoc = mlir::FileLineColLoc::get(mlirCtx, fileName, line, 1);
  return mlir::Location(fileLineColLoc);
}

auto matmulFunc(mlir::MLIRContext *ctx) {
  auto location = mkLoc(ctx, __LINE__);
  auto type = mlir::FunctionType::get(ctx, {}, {});
  auto func_name = "matmul";
  auto sym_name = mlir::StringAttr::get(ctx, func_name);
  auto sym_attr = mlir::NamedAttribute(llvm::StringRef("sym_name"), sym_name);
  auto attrs1 = {sym_attr};
  auto attrs = mlir::ArrayRef<mlir::NamedAttribute>(attrs1);
  auto funcOp = mlir::func::FuncOp::create(location, func_name, type, attrs);
  return funcOp;
}

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  mlir::sdy::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  auto threadingEnabled = mlir::MLIRContext::Threading::DISABLED;
  auto ctx = mlir::MLIRContext(registry, threadingEnabled);
  ctx.loadAllAvailableDialects();
  auto moduleOp = mlir::ModuleOp::create(mkLoc(&ctx, __LINE__), "main");
  moduleOp.getBody()->push_back(matmulFunc(&ctx));
  opPrint(moduleOp);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SDY Builder", registry));
}
