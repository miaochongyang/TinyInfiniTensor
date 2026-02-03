#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

    // 1. 找到需要优化的 Matmul 算子，注意使用 Operator 类型 (shared_ptr)
    Operator matmulOp = nullptr;
    for (auto &op : ops) {
        if (op->getOpType() == OpType::MatMul) {
            matmulOp = op;
            break;
        }
    }

    if (!matmulOp) return;
    auto matmul = as<MatmulObj>(matmulOp);

    // 2. 追踪新的输入源并计算属性
    Tensor newInputs[2];
    for (int i = 0; i < 2; ++i) {
        auto currentTensor = matmul->getInputs(i);
        auto prevOp = currentTensor->getSource();

        while (prevOp && prevOp->getOpType() == OpType::Transpose) {
            auto trans = as<TransposeObj>(prevOp);
            auto perm = trans->getPermute();
            int r = perm.size();

            if (r >= 2 && perm[r-1] == r-2 && perm[r-2] == r-1) {
                if (i == 0) matmul->setTransA(!matmul->getTransA());
                else matmul->setTransB(!matmul->getTransB());

                currentTensor = trans->getInputs(0);
                prevOp = currentTensor->getSource();
            } else {
                break;
            }
        }
        newInputs[i] = currentTensor;
    }

    // 3. 修改连接关系
    for (int i = 0; i < 2; ++i) {
        auto oldInput = matmul->getInputs(i);
        auto newInput = newInputs[i];

        if (oldInput != newInput) {
            // 清理新输入 (如 i1, i2) 的 targets 中指向旧 Transpose 的引用
            // 必须拷贝一份 targets 避免在 removeTarget 时导致迭代器失效
            auto targets = newInput->getTargets();
            for (auto& tgt : targets) {
                if (tgt->getOpType() == OpType::Transpose) {
                    newInput->removeTarget(tgt);
                }
            }

            // 建立新连接：让 i1/i2 指向 Matmul，Matmul 指向 i1/i2
            newInput->addTarget(matmulOp); // 这里使用 matmulOp (shared_ptr)
            matmul->inputs[i] = newInput;
        }
    }

    // 4. 重建全局列表 (垃圾回收)

    // 只保留 Matmul 算子
    OpVec newOps;
    newOps.push_back(matmulOp);
    ops = std::move(newOps);

    // 只保留 Matmul 涉及的 Tensor (输入和输出)
    TensorVec newTensors;
    for (auto &t : tensors) {
        bool keep = false;
        for (auto &in : matmul->getInputs()) {
            if (in == t) { keep = true; break; }
        }
        for (auto &out : matmul->getOutputs()) {
            if (out == t) { keep = true; break; }
        }

        if (keep) {
            newTensors.push_back(t);
        }
    }
    tensors = std::move(newTensors);

    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        vector<size_t> offsets;
        for (auto &tensor : tensors) {
            offsets.push_back(allocator.alloc(tensor->getBytes()));
        }

        void *heapPtr = allocator.getPtr();

        for (size_t i = 0; i < tensors.size(); ++i) {
            char *tensorPtr = (char *)heapPtr + offsets[i];
            tensors[i]->setDataBlob(make_ref<BlobObj>(runtime, tensorPtr));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini