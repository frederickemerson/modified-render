template <typename T>
CircularBuffer<T>::CircularBuffer(int size) : nextFreePtr(0), numOfElems(0), size(size)
{
    internalArr = new T[size];
}

template <typename T>
CircularBuffer<T>::CircularBuffer(CircularBuffer<T>&& cBuff) : internalArr(cBuff.internalArr)
{
    cBuff.internalArr = nullptr;
}

template <typename T>
CircularBuffer<T>& CircularBuffer<T>::operator=(CircularBuffer<T>&& cBuff) :
    internalArr(cBuff.internalArr)
{
    cBuff.internalArr = nullptr;
}

template <typename T>
CircularBuffer<T>::~CircularBuffer()
{
    delete[] internalArr;
}

template <typename T>
int CircularBuffer<T>::getSize()
{
    return size;
}

template <typename T>
int CircularBuffer<T>::getNumberOfElements()
{
    return numOfElems;
}

template <typename T>
void CircularBuffer<T>::push_back(T elem)
{
    *(internalArr + nextFreePtr) = elem;
    nextFreePtr = (nextFreePtr + 1) % size;
    if (numOfElems < size)
    {
        numOfElems++;
    }
}

template <typename T>
const T& CircularBuffer<T>::at(int index)
{
    // Check for boundaries
    if (index <= -numOfElems || index > 0)
    {
        char errorMsg[125];
        sprintf(errorMsg, "Index %d out-of-range in CircularBuffer with size %d, "
            "and number of elements %d.", index, size, numOfElems);
        throw std::out_of_range(errorMsg);
    }

    return *(internalArr + ((nextFreePtr - 1 + index + size) % size));
}