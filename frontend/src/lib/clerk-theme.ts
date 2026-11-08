// Styles Clerk's sign-in and register cards to match docs/design/screens/sign-in.png and register.png.
export const clerkAppearance = {
  variables: {
    colorPrimary: "#16a34a",
    colorForeground: "#0f172a",
    colorMutedForeground: "#64748b",
    colorBackground: "#ffffff",
    colorInput: "#ffffff",
    colorInputForeground: "#0f172a",
    colorNeutral: "#0f172a",
    colorDanger: "#b3261e",
    fontFamily: "var(--font-inter)",
    borderRadius: "0.5rem",
  },
  elements: {
    cardBox: "rounded-[18px] border border-line shadow-[0_16px_40px_rgb(20_83_45_/_0.08)]",
    headerTitle: "font-serif text-3xl text-brand-green italic",
    headerSubtitle: "text-muted",
    formFieldLabel: "label text-brand-green",
    formButtonPrimary: "label h-12 font-bold normal-case",
    socialButtonsBlockButton: "h-11",
    footerActionLink: "font-semibold text-brand-green underline underline-offset-4",
  },
};

export const clerkLocalization = {
  signIn: {
    start: { title: "Welcome back", subtitle: "Sign in to see your scans and reports." },
  },
  signUp: {
    start: { title: "Create an account", subtitle: "Your scans and reports stay private to your account." },
  },
};