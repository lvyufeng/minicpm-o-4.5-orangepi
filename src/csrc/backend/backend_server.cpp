// Backend WebSocket server stub - implements the protocol described in
// MiniCPM-o-Demo/docs/backend-protocol/
//
// This server accepts connections on /backend and implements the four primitives:
// - init: Initialize session
// - push: Push input events (text/audio/image frames)
// - pull: Pull output events (text/audio/end-of-turn)
// - unary: Single request-response

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::cout << "MiniCPM-O Backend Server (stub)" << std::endl;
    std::cout << "TODO: Implement WebSocket server with protocol from MiniCPM-o-Demo" << std::endl;
    std::cout << "      - /backend WebSocket endpoint" << std::endl;
    std::cout << "      - init/push/pull/unary message handling" << std::endl;
    std::cout << "      - Integration with inference engine" << std::endl;
    return 0;
}
