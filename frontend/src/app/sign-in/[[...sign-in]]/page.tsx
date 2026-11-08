import { SignIn } from "@clerk/nextjs";
import type { Metadata } from "next";

import { AuthCardPage } from "@/components/auth-card-page";

export const metadata: Metadata = { title: "Sign in" };

export default function SignInPage() {
  return (
    <AuthCardPage>
      <SignIn />
    </AuthCardPage>
  );
}