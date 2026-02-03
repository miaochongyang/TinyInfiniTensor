#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================

        auto shapeA = inputs[0]->getDims();
        auto shapeB = inputs[1]->getDims();
        int rankA = shapeA.size();
        int rankB = shapeB.size();

        int dimA1 = shapeA[rankA - 2];
        int dimA2 = shapeA[rankA - 1];
        int m = transA ? dimA2 : dimA1;
        int kA = transA ? dimA1 : dimA2;

        int dimB1 = shapeB[rankB - 2];
        int dimB2 = shapeB[rankB - 1];
        int kB = transB ? dimB2 : dimB1;
        int n = transB ? dimB1 : dimB2;

        IT_ASSERT(kA == kB);

        Shape batchA(shapeA.begin(), shapeA.end() - 2);
        Shape batchB(shapeB.begin(), shapeB.end() - 2);

        Shape outShape = infer_broadcast(batchA, batchB);

        outShape.push_back(m);
        outShape.push_back(n);

        return {{outShape}};
    }

} // namespace infini