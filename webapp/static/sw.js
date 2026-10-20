// Service worker: caches the app shell so the PWA opens offline and installs to the home screen.
// API calls are never cached: every analysis must come from the server.

const CACHE = "chart-analyzer-v1";
const SHELL = [
  "./", "index.html", "styles.css", "app.js", "manifest.json",
  "icons/icon-180.png", "icons/icon-192.png", "icons/icon-512.png", "icons/icon-maskable-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin || url.pathname.includes("/api/")) {
    return; // straight to the network
  }
  // Network first (so updates arrive), cache as the offline fallback.
  event.respondWith(
    fetch(event.request)
      .then((res) => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(event.request, copy));
        }
        return res;
      })
      .catch(() => caches.match(event.request).then((hit) => hit || caches.match("index.html"))),
  );
});
