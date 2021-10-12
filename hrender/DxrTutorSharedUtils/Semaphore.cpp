#include "Semaphore.h"

Semaphore::Semaphore(bool initialValue): isReady(initialValue) {}

inline void Semaphore::signal()
{
    {   // extra scope to ensure the lifetime of the mutex
        std::lock_guard<std::mutex> lock(mutex);
        isReady = true;
    }
    // condVar notify_one is thread-safe, the lock
    // does not need to be held for notification
    condVar.notify_one();
}


inline void Semaphore::wait()
{
    // acquire the mutex using std::unique_lock
    std::unique_lock<std::mutex> lock(mutex);
    // wait/block until this->isReady is true
    // (mutex will be atomically released and reacquired)
    condVar.wait(lock, [this](){ return this->isReady; });
    // set isReady to false
    isReady = false;
    // mutex is unlocked at the end of the scope
}
