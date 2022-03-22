// Implementation of a circular buffer

#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#include <memory>
#include <stdexcept>
#include <vector>

template <typename T>
class CircularBuffer {
public:
    CircularBuffer(int size);
    CircularBuffer(const CircularBuffer<T>& cBuff) = delete;
    CircularBuffer<T>& operator=(const CircularBuffer<T>& cBuff) = delete;
    CircularBuffer(CircularBuffer<T>&& cBuff);
    CircularBuffer<T>& operator=(CircularBuffer<T>&& cBuff);
    ~CircularBuffer();

    // Returns the size of this buffer
    int getSize();

    // Returns the number of elements in this buffer
    int getNumberOfElements();

    // Adds a new element to the buffer
    void push_back(T elem);

    // Access the element at the index specified
    // Valid range of indexes are [1 - size, 0]
    // 
    // For example, circularBuffer.at(0) will access the
    // last element added to the buffer.
    // 
    // circularBuffer.at(-1) will access the second-last
    // element added to the buffer, given that it exists
    // (more than 1 element has been added previously,
    // and size of the buffer is greater than 1).
    const T& at(int index);

private:
    int nextFreePtr = 0;
    int numOfElems = 0;
    int size = 0;
    T* internalArr = nullptr;
};

#include "CircularBuffer.hpp"

#endif CIRCULAR_BUFFER_H