#include <vector>

class PlsWork {
public:
	static void* compress(size_t* bytes, std::vector<uint8_t> uncompressed_data);
};