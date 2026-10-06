import CoreML
import Foundation

@main
struct ComputePlan {
    static func main() async throws {
        guard CommandLine.arguments.count == 4 else {
            throw NSError(domain: "qwen3-plan", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "usage: plan model.mlmodelc cpu|gpu|ane function"])
        }
        let configuration = MLModelConfiguration()
        switch CommandLine.arguments[2] {
        case "cpu": configuration.computeUnits = .cpuOnly
        case "gpu": configuration.computeUnits = .cpuAndGPU
        case "ane": configuration.computeUnits = .cpuAndNeuralEngine
        default: throw NSError(domain: "qwen3-plan", code: 2)
        }
        configuration.functionName = CommandLine.arguments[3]
        let plan = try await MLComputePlan.load(
            contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]), configuration: configuration)
        guard case .program(let program) = plan.modelStructure,
              let function = program.functions[CommandLine.arguments[3]] else {
            throw NSError(domain: "qwen3-plan", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "requested ML Program function is missing"])
        }
        var operations: [[String: String]] = []
        func visit(_ block: MLModelStructure.Program.Block) {
            for operation in block.operations {
                var preferred = "unknown"
                if let usage = plan.deviceUsage(for: operation) {
                    switch usage.preferred {
                    case .cpu: preferred = "MLCPUComputeDevice"
                    case .gpu: preferred = "MLGPUComputeDevice"
                    case .neuralEngine: preferred = "MLNeuralEngineComputeDevice"
                    @unknown default: preferred = "unknown"
                    }
                }
                operations.append(["operator": operation.operatorName, "preferred": preferred])
                for child in operation.blocks { visit(child) }
            }
        }
        visit(function.block)
        var counts: [String: Int] = [:]
        for operation in operations { counts[operation["preferred"]!, default: 0] += 1 }
        let result: [String: Any] = ["function": CommandLine.arguments[3],
                                   "preferred_operation_counts": counts, "operations": operations]
        let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }
}
