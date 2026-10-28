
import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const model = new Supabase.ai.Session("gte-small");

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok");
  }

  if (req.method !== "POST") {
    return Response.json({ error: "POST required" }, { status: 405 });
  }

  try {
    const { texts } = await req.json();

    if (
      !Array.isArray(texts) ||
      texts.length < 1 ||
      texts.length > 16 ||
      !texts.every(
        (t: unknown) =>
          typeof t === "string" && t.trim().length > 0
      )
    ) {
      return Response.json(
        { error: "texts must contain 1-16 non-empty strings" },
        { status: 400 }
      );
    }

    const embeddings = [];

    for (const text of texts) {
      const embedding = await model.run(text, {
        mean_pool: true,
        normalize: true,
      });

      if (!Array.isArray(embedding) || embedding.length !== 384) {
        throw new Error("Unexpected embedding dimensions");
      }

      embeddings.push(embedding);
    }

    return Response.json({
      model: "gte-small",
      dimensions: 384,
      embeddings,
    });
  } catch (error) {
    console.error("Embedding error:", error);
    return Response.json(
      { error: "Embedding generation failed" },
      { status: 500 }
    );
  }
});
