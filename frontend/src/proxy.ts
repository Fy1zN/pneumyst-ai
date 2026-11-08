import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

// Pages that show or create a user's scans. Everything else (home, sign-in, register) stays public.
const isPrivate = createRouteMatcher(["/scan(.*)", "/history(.*)", "/results(.*)"]);

export default clerkMiddleware(async (auth, request) => {
  if (isPrivate(request)) await auth.protect();
});

export const config = {
  matcher: [
    // Skip Next.js internals and static files, unless they are in the search params.
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};