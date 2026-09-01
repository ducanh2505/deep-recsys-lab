const SOCIAL_IMAGE_TOKEN = "__SITE_ORIGIN__/og.png";

export default {
  async fetch(request, env) {
    const response = await env.ASSETS.fetch(request);
    const contentType = response.headers.get("content-type") || "";
    if (request.method === "HEAD" || !contentType.includes("text/html")) {
      return response;
    }

    const html = await response.text();
    const headers = new Headers(response.headers);
    headers.delete("content-length");
    const origin = new URL(request.url).origin;
    return new Response(html.replaceAll(SOCIAL_IMAGE_TOKEN, `${origin}/og.png`), {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  },
};
