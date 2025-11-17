module LIMPSBridge

using HTTP
using Sockets
using JSON3
using LinearAlgebra
using Statistics

# Try to detect a locally available AL_ULS module
const HAS_ALULS = let ok = false
    try
        @eval using AL_ULS
        ok = isdefined(@__MODULE__, :AL_ULS) && isdefined(AL_ULS, :optimize)
    catch
        ok = false
    end
    ok
end

struct CoherenceResult
    ghost_score::Float64
    avg_entropy::Float64
    avg_sensitivity::Float64
    non_idempotence::Float64
    non_commutativity::Float64
    non_associativity::Float64
    hotspots::Vector{String}
end

kfp_smooth(x; α=0.35) = α * x + (1 - α) * tanh(x)

function compute_coherence(result)
    fns = get(result, "functions", Any[])
    ent = Float64[]
    sens = Float64[]
    nonidem = Float64[]
    noncomm = Float64[]
    nonassoc = Float64[]
    hotspots = String[]

    for f in fns
        p = get(f, "probe", nothing)
        p === nothing && continue
        push!(ent, try parse(Float64, string(get(p, "entropy_bits", 0.0))) catch; 0.0 end)
        push!(sens, try parse(Float64, string(get(p, "sensitivity", 0.0))) catch; 0.0 end)
        idr = try parse(Float64, string(get(p, "idempotent_rate", 0.0))) catch; 0.0 end
        push!(nonidem, max(0.0, 1.0 - idr))
        cr = get(p, "commutative_rate", nothing)
        ar = get(p, "associative_rate", nothing)
        push!(noncomm, cr === nothing ? 0.0 : max(0.0, 1.0 - try parse(Float64, string(cr)) catch; 0.0 end))
        push!(nonassoc, ar === nothing ? 0.0 : max(0.0, 1.0 - try parse(Float64, string(ar)) catch; 0.0 end))

        # Fixed: Parse cr/ar before comparing to avoid type errors
        cr_val = cr === nothing ? nothing : try parse(Float64, string(cr)) catch; nothing end
        ar_val = ar === nothing ? nothing : try parse(Float64, string(ar)) catch; nothing end

        if (idr < 0.4) || ((cr_val !== nothing) && (cr_val < 0.5)) || ((ar_val !== nothing) && (ar_val < 0.5))
            push!(hotspots, String(get(f, "qualname", "")))
        end
    end

    μH = isempty(ent) ? 0.0 : mean(ent)
    μS = isempty(sens) ? 0.0 : mean(sens)
    μNid = isempty(nonidem) ? 0.0 : mean(nonidem)
    μNc = isempty(noncomm) ? 0.0 : mean(noncomm)
    μNa = isempty(nonassoc) ? 0.0 : mean(nonassoc)

    raw = 0.45 * μH / 8 + 0.25 * kfp_smooth(μS) + 0.2 * μNid + 0.05 * μNc + 0.05 * μNa
    ghost = 1 / (1 + exp(-4 * (raw - 0.5)))
    return CoherenceResult(ghost, μH, μS, μNid, μNc, μNa, hotspots)
end

function result_to_json(cr::CoherenceResult)
    return JSON3.write(Dict(
        :ghost_score => cr.ghost_score,
        :avg_entropy => cr.avg_entropy,
        :avg_sensitivity => cr.avg_sensitivity,
        :non_idempotence => cr.non_idempotence,
        :non_commutativity => cr.non_commutativity,
        :non_associativity => cr.non_associativity,
        :hotspots => cr.hotspots,
    ))
end

function _to_matrix(A)
    # Fixed: Guard empty input to avoid reduce(vcat, …) on []
    isempty(A) && return zeros(Float64, 0, 0)
    rows = [Float64.(A[i]) for i in 1:length(A)]
    return reduce(vcat, (permutedims(r) for r in rows))
end

function run_al_uls(adj::Matrix{Float64}; kwargs...)
    if HAS_ALULS
        return AL_ULS.optimize(adj; kwargs...)
    else
        url = get(ENV, "AL_ULS_URL", "")
        isempty(url) && error("AL_ULS_URL not set and local AL_ULS not available")
        # Fixed: Merge kwargs into the top-level JSON instead of nested options
        payload = JSON3.write(merge(Dict(:adjacency => adj), Dict(kwargs)))
        resp = HTTP.post(string(url, "/optimize"), ["Content-Type" => "application/json"], payload)
        resp.status == 200 || error("al-ULS HTTP error: $(resp.status)")
        return JSON3.read(String(resp.body))
    end
end

function handle(req::HTTP.Request)
    target = String(req.target)
    if req.method == "GET" && target == "/health"
        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON3.write(Dict("ok" => true)))
    elseif req.method == "POST" && target == "/coherence"
        payload = JSON3.read(String(req.body))
        cr = compute_coherence(payload)
        return HTTP.Response(200, ["Content-Type" => "application/json"], result_to_json(cr))
    elseif req.method == "POST" && target == "/optimize"
        payload = JSON3.read(String(req.body))
        adj_json = get(payload, "adjacency", nothing)
        adj_json === nothing && return HTTP.Response(400, ["Content-Type" => "application/json"], JSON3.write(Dict("error" => "missing adjacency")))
        adj = _to_matrix(adj_json)
        mode = get(payload, "mode", "kfp")
        beta = try parse(Float64, string(get(payload, "beta", 0.8))) catch; 0.8 end
        adj2 = kfp_smooth.(adj .* beta)
        result = run_al_uls(adj2; mode=mode, beta=beta)

        # Fixed: Include labels in response if provided for tracing
        labels = get(payload, "labels", nothing)
        response_data = Dict(:ok => true, :mode => mode, :beta => beta, :n => size(adj2, 1), :result => result)
        if labels !== nothing
            response_data[:labels] = labels
        end

        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON3.write(response_data))
    else
        return HTTP.Response(404, ["Content-Type" => "application/json"], JSON3.write(Dict("error" => "not found")))
    end
end

function serve(; host::AbstractString = "0.0.0.0", port::Integer = 8099)
    @info "LIMPSBridge listening" host=host port=port
    HTTP.serve(handle, host, port)
end

end # module
