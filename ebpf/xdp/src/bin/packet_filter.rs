#![no_std]
#![no_main]

use core::mem;

use aya_bpf::bindings::xdp_action;
use aya_bpf::macros::{map, xdp};
use aya_bpf::maps::HashMap;
use aya_bpf::programs::XdpContext;
use aya_log_ebpf::info;
use common::IpPortKey;
use memoffset::offset_of;
use network_types::eth::{EthHdr, EtherType};
use network_types::ip::{IpProto, Ipv4Hdr};
use network_types::tcp::TcpHdr;
use network_types::udp::UdpHdr;

type XdpAction = xdp_action::Type;

#[map]
static BLOCK_LIST: HashMap<IpPortKey, u32> = HashMap::<IpPortKey, u32>::with_max_entries(1024, 0);

#[xdp]
pub fn xdp_packet_filter(ctx: XdpContext) -> u32 {
    match unsafe { filter(ctx) } {
        Ok(ret) => ret,
        Err(_) => xdp_action::XDP_ABORTED,
    }
}

unsafe fn filter(ctx: XdpContext) -> Result<u32, ()> {
    let h_proto = unsafe { *ptr_at::<EtherType>(&ctx, offset_of!(EthHdr, ether_type))? };
    match h_proto {
        EtherType::Ipv4 => process_ipv4(&ctx),
        _ => Ok(xdp_action::XDP_PASS),
    }
}

fn process_ipv4(ctx: &XdpContext) -> Result<XdpAction, ()> {
    let proto = unsafe { *ptr_at::<IpProto>(&ctx, EthHdr::LEN + offset_of!(Ipv4Hdr, proto))? };
    let ip =
        u32::from_be_bytes(unsafe { *ptr_at(&ctx, EthHdr::LEN + offset_of!(Ipv4Hdr, src_addr))? });

    info!(ctx, "received a packet from {:i}", ip);

    let port = match proto {
        IpProto::Tcp => u16::from_be_bytes(unsafe {
            *ptr_at(&ctx, EthHdr::LEN + Ipv4Hdr::LEN + offset_of!(TcpHdr, dest))?
        }),
        IpProto::Udp => u16::from_be_bytes(unsafe {
            *ptr_at(&ctx, EthHdr::LEN + Ipv4Hdr::LEN + offset_of!(UdpHdr, dest))?
        }),
        _ => {
            return Ok(xdp_action::XDP_PASS);
        },
    };

    if not_allowed(ip, port) {
        Ok(xdp_action::XDP_DROP)
    } else {
        Ok(xdp_action::XDP_PASS)
    }
}

// Before any data access, the verifier requires us to do a bound check.
fn ptr_at<T>(ctx: &XdpContext, offset: usize) -> Result<*const T, ()> {
    let start = ctx.data();
    let end = ctx.data_end();
    let len = mem::size_of::<T>();

    if start + offset + len > end {
        return Err(());
    }

    Ok((start + offset) as *const T)
}

fn not_allowed(ip: u32, port: u16) -> bool {
    unsafe { BLOCK_LIST.get(&IpPortKey { ip, port: port as u32 }).is_some() }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe { core::hint::unreachable_unchecked() }
}
