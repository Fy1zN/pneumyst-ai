import { ClerkProvider } from "@clerk/nextjs";
import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Montserrat, Playfair_Display } from "next/font/google";
import { Suspense } from "react";

import { SiteFooter } from "@/components/site-footer";
import { SiteNav, SiteNavFallback } from "@/components/site-nav";
import { clerkAppearance, clerkLocalization } from "@/lib/clerk-theme";
import "./globals.css";

// next/font serves these from our own site, so visitors' browsers never contact Google.
const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const playfair = Playfair_Display({ variable: "--font-playfair", subsets: ["latin"], weight: "700", style: "italic" });
const jetbrains = JetBrains_Mono({ variable: "--font-jetbrains", subsets: ["latin"], weight: ["500", "700"] });
const montserrat = Montserrat({ variable: "--font-montserrat", subsets: ["latin"], weight: ["500", "800"] });

export const metadata: Metadata = {
  title: { default: "PNEUMYST AI", template: "%s · PNEUMYST AI" },
  description:
    "Explainable chest X-ray screening for tuberculosis and 14 other findings. Research prototype, not for clinical use.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <ClerkProvider appearance={clerkAppearance} localization={clerkLocalization}>
      <html
        lang="en"
        className={`${inter.variable} ${playfair.variable} ${jetbrains.variable} ${montserrat.variable} h-full antialiased`}
      >
        <body className="flex min-h-full flex-col font-sans">
          {/* The nav reads the current URL, which some pages only know at request time. */}
          <Suspense fallback={<SiteNavFallback />}>
            <SiteNav />
          </Suspense>
          {children}
          <SiteFooter />
        </body>
      </html>
    </ClerkProvider>
  );
}