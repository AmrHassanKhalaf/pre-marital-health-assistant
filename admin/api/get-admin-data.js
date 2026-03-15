module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { hfSpaceUrl, adminKey } = req.body || {};

    if (!hfSpaceUrl || !adminKey) {
      return res.status(400).json({ error: "Missing hfSpaceUrl or adminKey" });
    }

    const baseUrl = String(hfSpaceUrl).replace(/\/+$/, "");

    const callResp = await fetch(`${baseUrl}/gradio_api/call/get_admin_data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: [adminKey] }),
    });

    if (!callResp.ok) {
      return res.status(callResp.status).json({ error: `HF call error (${callResp.status})` });
    }

    const callJson = await callResp.json();
    const eventId = callJson.event_id;

    if (!eventId) {
      return res.status(502).json({ error: "Missing event_id from HF response" });
    }

    const sseResp = await fetch(`${baseUrl}/gradio_api/call/get_admin_data/${eventId}`);
    if (!sseResp.ok) {
      return res.status(sseResp.status).json({ error: `HF SSE error (${sseResp.status})` });
    }

    const sseText = await sseResp.text();
    let output = "";

    const sseLines = sseText.split("\n");
    for (let i = 0; i < sseLines.length; i++) {
      if (sseLines[i].startsWith("event: complete") && i + 1 < sseLines.length) {
        const dataLine = sseLines[i + 1];
        if (dataLine.startsWith("data: ")) {
          const parsed = JSON.parse(dataLine.slice(6));
          output = parsed[0];
        }
      }
    }

    if (!output) {
      return res.status(502).json({ error: "No output returned from HF stream" });
    }

    const data = JSON.parse(output);
    return res.status(200).json(data);
  } catch (error) {
    return res.status(500).json({ error: error.message || "Internal server error" });
  }
};
