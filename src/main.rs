use anyhow::Result;
use pixels::wgpu::Color;
use pixels::{Pixels, PixelsBuilder, SurfaceTexture};
use rand::Rng;
use std::f64::consts::PI;
use std::time::{Duration, Instant};
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalPosition};
use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, StartCause, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

const FPS: u64 = 60;

const GRAVITATION: f64 = 0.1;

struct Star {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    m: f64,
}

struct World {
    stars: Vec<Star>,
    camera_x: f64,
    camera_y: f64,
    camera_zoom: f64,
}

impl World {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut stars = Vec::new();

        for _ in 0..1400 {
            let r1 = rng.gen_range(0.0..10000.0);
            let t1 = rng.gen_range(0.0..2.0 * PI);
            let r2 = rng.gen_range(-0.0..10.0);
            let t2 = rng.gen_range(0.0..2.0 * PI);
            stars.push(Star {
                x: r1 * t1.cos(),
                y: r1 * t1.sin(),
                vx: r2 * t2.cos(),
                vy: r2 * t2.sin(),
                m: rng.gen_range(0.0..1000.0),
            });
        }

        Self {
            stars,
            camera_x: 0.0,
            camera_y: 0.0,
            camera_zoom: 0.2,
        }
    }

    fn update(&mut self) {
        for i in 0..self.stars.len() {
            let (before, [target, after @ ..]) = self.stars.split_at_mut(i) else {unreachable!()};

            let (ax, ay) = before
                .iter()
                .chain(after.iter())
                .fold((0.0, 0.0), |(ax, ay), src| {
                    let x_diff = src.x - target.x;
                    let y_diff = src.y - target.y;
                    let c = GRAVITATION * src.m / (x_diff.powi(2) + y_diff.powi(2)).powf(1.5);
                    (ax + c * x_diff, ay + c * y_diff)
                });

            target.vx += ax;
            target.vy += ay;
            target.x += target.vx;
            target.y += target.vy;
        }
    }

    fn camera_zoom(&mut self, x: f64) {
        self.camera_zoom *= (1.1f64).powf(x);
    }

    fn camera_move(&mut self, vx: f64, vy: f64) {
        self.camera_x += vx / self.camera_zoom;
        self.camera_y += vy / self.camera_zoom;
    }

    fn draw(&self, frame: &mut [u8], frame_size: LogicalSize<u32>) {
        for val in frame.iter_mut() {
            *val = 0;
        }
        for star in &self.stars {
            let radius = star.m.sqrt();
            draw_circle(
                ((star.x - self.camera_x) * self.camera_zoom) as i32
                    + (frame_size.width / 2) as i32,
                ((star.y - self.camera_y) * self.camera_zoom) as i32
                    + (frame_size.height / 2) as i32,
                (radius * self.camera_zoom) as u32,
                frame,
                frame_size.width,
                frame_size.height,
            );
        }
    }
}

fn draw_circle(cx: i32, cy: i32, r: u32, frame: &mut [u8], frame_width: u32, frame_height: u32) {
    let mut draw_horizontal_line = |x_start: i32, x_end: i32, y: i32| {
        let base = if (0..frame_height as i32).contains(&(y + cy)) {
            ((y + cy) as u32 * frame_width) as usize
        } else {
            return;
        };

        let start = base + (x_start + cx).max(0).min(frame_width as i32) as usize;
        let end = base + (x_end + cx + 1).max(0).min(frame_width as i32) as usize;

        for pixel in frame[4 * start..4 * end].chunks_exact_mut(4) {
            pixel.copy_from_slice(&[0x5e, 0x48, 0xe8, 0xff]);
        }
    };

    let r = r as i32;
    draw_horizontal_line(-r, r, 0);

    let mut x = r;
    let mut y = 0;

    while {
        x = {
            let p = ((y + 1) * (y + 1) + (x - 1) * (x - 1)).abs_diff(r * r);
            let q = ((y + 1) * (y + 1) + x * x).abs_diff(r * r);
            if p < q {
                draw_horizontal_line(-y, y, x);
                draw_horizontal_line(-y, y, -x);
                x - 1
            } else {
                x
            }
        };
        y += 1;
        y < x
    } {
        draw_horizontal_line(-x, x, y);
        draw_horizontal_line(-x, x, -y);
    }

    if y == x {
        draw_horizontal_line(-x, x, y);
        draw_horizontal_line(-x, x, -y);
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("rsstars")
        .build(&event_loop)
        .unwrap();

    let mut pixels = {
        let size = window.inner_size();
        let surface_texture = SurfaceTexture::new(size.width, size.height, &window);
        PixelsBuilder::new(size.width, size.height, surface_texture)
            .clear_color(Color {
                r: 0.0,
                g: 0.001,
                b: 0.02,
                a: 1.0,
            })
            .build()?
    };

    let mut world = World::new();

    let mut context = Context::new(&window, FPS);

    event_loop.run(move |event, _, control_flow| {
        match event_handler(event, &mut context, &mut world, &window, &mut pixels) {
            Ok(new_control_flow) => {
                if let Some(new_control_flow) = new_control_flow {
                    *control_flow = new_control_flow;
                }
            }
            Err(err) => {
                eprintln!("{err:?}");
                control_flow.set_exit();
            }
        }
    });
}

#[derive(Default)]
struct CursorPosition {
    prev: Option<PhysicalPosition<f64>>,
    now: Option<PhysicalPosition<f64>>,
}

struct Context {
    delta: Duration,
    scale_factor: f64,
    frame_size: LogicalSize<u32>,
    waiting: bool,
    cursor_pos: CursorPosition,
    mouse_pressing: bool,
}

impl Context {
    fn new(window: &Window, fps: u64) -> Self {
        let scale_factor = window.scale_factor();
        Context {
            delta: Duration::from_nanos(1000_000_000 / fps),
            scale_factor,
            frame_size: window.inner_size().to_logical(scale_factor),
            waiting: false,
            cursor_pos: CursorPosition::default(),
            mouse_pressing: false,
        }
    }
}

fn event_handler<T>(
    event: Event<'_, T>,
    context: &mut Context,
    world: &mut World,
    window: &Window,
    pixels: &mut Pixels,
) -> Result<Option<ControlFlow>> {
    match event {
        Event::NewEvents(start_cause) => match start_cause {
            StartCause::WaitCancelled {
                requested_resume, ..
            } => {
                context.waiting = true;
                if let Some(waiting_until) = requested_resume {
                    Ok(Some(ControlFlow::WaitUntil(waiting_until)))
                } else {
                    Ok(Some(ControlFlow::Wait))
                }
            }

            _ => {
                context.waiting = false;
                Ok(Some(ControlFlow::WaitUntil(Instant::now() + context.delta)))
            }
        },

        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => Ok(Some(ControlFlow::Exit)),

            WindowEvent::ScaleFactorChanged {
                scale_factor: new_scale_factor,
                ..
            } => {
                context.scale_factor = new_scale_factor;
                Ok(None)
            }

            WindowEvent::Resized(size) => {
                pixels.resize_surface(size.width, size.height)?;
                context.frame_size = size.to_logical(context.scale_factor);
                pixels.resize_buffer(context.frame_size.width, context.frame_size.height)?;
                Ok(None)
            }

            WindowEvent::CursorMoved { position, .. } => {
                context.cursor_pos.prev = context.cursor_pos.now;
                context.cursor_pos.now = Some(position);

                if context.mouse_pressing {
                    if let CursorPosition {
                        prev: Some(physical_prev),
                        now: Some(physical_now),
                    } = context.cursor_pos
                    {
                        let prev: LogicalPosition<f64> =
                            physical_prev.to_logical(context.scale_factor);
                        let now: LogicalPosition<f64> =
                            physical_now.to_logical(context.scale_factor);

                        world.camera_move(prev.x - now.x, prev.y - now.y);
                    }
                }

                Ok(None)
            }

            WindowEvent::CursorLeft { .. } => {
                context.cursor_pos.now = None;
                context.cursor_pos.prev = None;
                Ok(None)
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    match state {
                        ElementState::Pressed => {
                            context.mouse_pressing = true;
                        }
                        ElementState::Released => {
                            context.mouse_pressing = false;
                        }
                    }
                }
                Ok(None)
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let mut vertical_delta = 0.0;

                match delta {
                    MouseScrollDelta::LineDelta(.., rows) => vertical_delta += rows as f64,
                    MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => vertical_delta += y,
                }

                world.camera_zoom(vertical_delta as f64);
                Ok(None)
            }
            _ => Ok(None),
        },

        Event::MainEventsCleared => {
            if !context.waiting {
                world.update();
                window.request_redraw();
            }
            Ok(None)
        }

        Event::RedrawRequested(_) => {
            world.draw(pixels.frame_mut(), context.frame_size);
            pixels.render()?;
            Ok(None)
        }

        _ => Ok(None),
    }
}
