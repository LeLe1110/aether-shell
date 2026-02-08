// Minimal SW — no caching, just pass through all requests.
// Clears any old caches on activation.

self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(keys.map((k) => caches.delete(k)));
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    // Pass through everything — no SW interception
    return;
});
