#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================

        size_t addr;  // return address
        // search free blocks
        for (auto &blocksPair: freeBlocks) {
            // if get free block
            if (blocksPair.second >= size) {
                // mark addr
                addr = blocksPair.first;
                // calculate remain free size
                blocksPair.second -= size;
                // if blocks is full, remove it
                if (blocksPair.second == 0) {
                    freeBlocks.erase(blocksPair.first);
                } else {
                    // else remake start position
                    auto newAddr = blocksPair.first + size;
                    auto newSize = blocksPair.second;
                    freeBlocks.erase(blocksPair.first);
                    freeBlocks[newAddr] = newSize;
                }
                return addr;
            }
        }
        // no find the free blocks;
        addr = peak;
        peak += size;

        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        // if end is free
        for (auto &blocksPair: freeBlocks) {
            auto &blockAddr = blocksPair.first;
            auto &blockSize = blocksPair.second;

            if (addr + size == blockAddr) {
                // merge 2 blocks
                size += blockSize;
                freeBlocks.erase(blockAddr);
                break;
            }
        }

        // if pre is free
        int flag = 0;
        for (auto &blocksPair: freeBlocks) {
            auto &blockAddr = blocksPair.first;
            auto &blockSize = blocksPair.second;

            if (blockAddr + blockSize == addr) {
                // merge 2 blocks
                blockSize += size;
                flag = 1;
                break;
            }
        }

        if (addr + size == peak) {
            peak -= size;
            if (flag == 1) {
                freeBlocks.erase(addr);
            }
            return;
        }

        if (flag == 0) {
            freeBlocks[addr] = size;
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
