# Recommended Tech Stack for VideoInquire

Based on an in-depth analysis of open-source VideoAsk alternatives (such as FormOnce and fk-videoAsk) and a review of the broader open-source ecosystem for video processing and form building, this document outlines the recommended technology stack for building **VideoInquire**—a self-hosted, white-label alternative to VideoAsk.

## 1. Frontend Framework Recommendation

**Recommendation: Next.js (React) with Tailwind CSS and shadcn/ui**

Next.js has emerged as the industry standard for building robust, production-ready React applications. The analysis of FormOnce demonstrated the effectiveness of the T3 Stack (Next.js, Prisma, tRPC, Tailwind) for building complex, interactive form builders. 

*   **Why Next.js:** It provides excellent server-side rendering (SSR) capabilities, which is crucial for the performance and SEO of the embeddable widgets and public-facing forms. The App Router in Next.js 14+ simplifies API route creation and server-side logic.
*   **Why Tailwind CSS & shadcn/ui:** Tailwind allows for rapid, utility-first styling, which is essential for creating a white-label product where themes and colors need to be easily customizable. `shadcn/ui` provides accessible, unstyled components that can be fully customized to match any brand identity, avoiding the "cookie-cutter" look of traditional component libraries.

## 2. Video Recording Technology

**Recommendation: Native MediaRecorder API with `react-media-recorder`**

For capturing video and audio directly in the browser, the native HTML5 MediaRecorder API is the most performant and lightweight approach.

*   **Implementation:** Instead of relying on heavy wrappers like RecordRTC (which is feature-rich but often overkill and bloated), using a lightweight React hook wrapper like `react-media-recorder` (or a custom hook inspired by the `fk-videoAsk` repository) provides fine-grained control over the recording state, media streams, and blob generation.
*   **Best Practices:** Ensure `getUserMedia` is configured with `facingMode: "user"` for mobile devices to default to the selfie camera. Implement robust error handling for permissions (camera/microphone access) as demonstrated in the analyzed repositories.

## 3. CDN / Storage Approach (Self-Hosted)

**Recommendation: MinIO (S3-Compatible Object Storage)**

Since VideoInquire must be deployable on the user's own infrastructure and not dependent on SaaS providers like AWS or BunnyCDN (which FormOnce uses), a self-hosted object storage solution is required.

*   **Why MinIO:** MinIO is a high-performance, open-source object storage server that is 100% compatible with the Amazon S3 API. This means you can use standard S3 SDKs (like the AWS SDK for JavaScript) in your application, but point them to your self-hosted MinIO instance.
*   **Upload Pipeline:** For handling large video files, implement chunked, resumable uploads using the **tus protocol** (via `tus-js-client` on the frontend and a tus-compatible server implementation). This ensures reliable uploads even on unstable mobile networks.

## 4. Video Processing Pipeline

**Recommendation: Server-Side FFmpeg via Docker / Go Microservice**

Video processing (transcoding, generating thumbnails, compressing) is computationally intensive and should not block the main Node.js application thread.

*   **Architecture:** Create a dedicated, scalable microservice for video processing. A common and highly efficient pattern is to use a Go-based service that wraps **FFmpeg**. 
*   **Workflow:** When a video is uploaded to MinIO, a webhook or message queue triggers the FFmpeg service. The service pulls the raw video, transcodes it into web-optimized formats (e.g., H.264/MP4 for broad compatibility, or HLS for adaptive streaming), generates a poster image, and uploads the processed assets back to MinIO.
*   **Alternative (Browser-side):** While `ffmpeg.wasm` allows for browser-side processing, it is generally not recommended for this use case due to high memory usage, lack of hardware acceleration on many devices, and poor battery performance on mobile. Server-side processing ensures a consistent user experience.

## 5. Authentication Method

**Recommendation: NextAuth.js (Auth.js) or self-hosted Supabase Auth**

*   **NextAuth.js:** If building a monolithic Next.js application, NextAuth.js is the most seamless integration. It supports credential-based logins (email/password) and OAuth providers, and integrates perfectly with Prisma.
*   **Supabase Auth:** If you prefer a Backend-as-a-Service (BaaS) approach that you can self-host via Docker, Supabase provides an excellent authentication module along with its PostgreSQL database. The `fk-videoAsk` repository successfully utilizes this pattern.

## 6. Database Recommendations

**Recommendation: PostgreSQL with Prisma ORM**

*   **Why PostgreSQL:** It is the most robust, open-source relational database, perfectly suited for handling complex relationships between users, workspaces, forms, questions, conditional logic rules, and video metadata.
*   **Why Prisma:** Prisma provides a highly developer-friendly, type-safe ORM that integrates flawlessly with Next.js and TypeScript. The schema definition is declarative and easy to maintain, as seen in the FormOnce repository.

## 7. Useful Open-Source Libraries and Components

To accelerate the development of VideoInquire, the following open-source libraries should be integrated:

### Form Building & Conditional Logic
*   **React Flow (`xyflow`):** An incredibly powerful library for building node-based UIs. This is highly recommended for building the visual "flow builder" where users can map out the branching logic of their video forms (e.g., "If user answers A, go to Video 2; if B, go to Video 3"). FormOnce uses this to great effect.
*   **SurveyJS (Reference):** While SurveyJS is a complete product, reviewing their open-source schema for conditional logic can provide valuable insights into structuring your own JSON schema for form branching.

### Embeddable Widget Framework
*   **Shadow DOM Isolation:** To ensure the VideoInquire widget can be embedded on *any* website without CSS conflicts, the widget must be encapsulated within a Shadow DOM. 
*   **Rollup / Vite:** Use Rollup (or Vite's library mode) to bundle the React widget into a single, self-contained IIFE (Immediately Invoked Function Expression) JavaScript file. This allows customers to install the widget with a single `<script>` tag. The `makerkit/react-embeddable-widget` repository provides an excellent boilerplate for this exact architecture.

### Video Playback
*   **Video.js:** If you need advanced playback features, custom controls, or HLS streaming support, Video.js is the industry standard open-source player. For simpler needs, the native HTML5 `<video>` element with custom React overlays (as seen in the `videoask_task_assignment` repo) is sufficient and lighter.
