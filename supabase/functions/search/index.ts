import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const model = new Supabase.ai.Session("gte-small");

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok");
  }

  if (req.method !== "POST") {
    return Response.json(
      { error: "POST required" },
      { status: 405 }
    );
  }

  try {
    const body = await req.json();
    const question = body?.question;
    const requestedCount = body?.match_count ?? 5;

    if (
      typeof question !== "string" ||
      question.trim().length === 0 ||
      question.length > 4000
    ) {
      return Response.json(
        { error: "question must be a non-empty string of at most 4000 characters" },
        { status: 400 }
      );
    }

    if (
      !Number.isInteger(requestedCount) ||
      requestedCount < 1 ||
      requestedCount > 10
    ) {
      return Response.json(
        { error: "match_count must be an integer from 1 to 10" },
        { status: 400 }
      );
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");

    if (!supabaseUrl || !serviceKey) {
      throw new Error("Required Supabase environment variables are unavailable");
    }

    const queryEmbedding = await model.run(question.trim(), {
      mean_pool: true,
      normalize: true,
    });

    if (!Array.isArray(queryEmbedding) || queryEmbedding.length !== 384) {
      throw new Error("Unexpected query embedding dimensions");
    }

    const rpcResponse = await fetch(
      `${supabaseUrl}/rest/v1/rpc/match_chunks`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "apikey": serviceKey,
          "Authorization": `Bearer ${serviceKey}`,
        },
        body: JSON.stringify({
          p_query_embedding:
            "[" + queryEmbedding.map((v: number) => Number(v).toString()).join(",") + "]",
          p_match_count: requestedCount,
          p_document_id: null,
        }),
      },
    );

    if (!rpcResponse.ok) {
      const detail = await rpcResponse.text();
      console.error("Vector search failed:", rpcResponse.status, detail);
      return Response.json(
        { error: "Vector search failed" },
        { status: 502 },
      );
    }

    const matches = await rpcResponse.json();

    const results = matches.map((match: {
      chunk_id: string;
      document_id: string;
      chunk_index: number;
      content: string;
      metadata: Record<string, unknown> | null;
      similarity: number;
    }) => ({
      chunk_id: match.chunk_id,
      document_id: match.document_id,
      chunk_index: match.chunk_index,
      content: match.content,
      similarity: match.similarity,
      source: {
        source_path: match.metadata?.source_path ?? null,
        source_name: match.metadata?.source_name ?? null,
      },
    }));

    return Response.json({
      question: question.trim(),
      model: "gte-small",
      result_count: results.length,
      results,
    });
  } catch (error) {
    console.error("Search function error:", error);
    return Response.json(
      { error: "Search failed" },
      { status: 500 },
    );
  }
});
